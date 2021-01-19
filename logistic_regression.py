import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# prepare data
bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)
# normalize
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
n_samples,n_features = X_train.shape

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
print(f'X_train.shape {X_train.shape} y_train.shape {y_train.shape}')
print(f'X_test.shape {X_test.shape} y_test.shape {y_test.shape}')

class LogisticRegression:
    def __init__(self,X,y,epochs,lmda=0.01,learning_rate=0.005):
        self.n_epochs = epochs
        self.lmda = lmda
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.n_samples,self.n_features = self.X.shape
        self.W = np.random.rand(self.n_features,1)
    def sigmoid(self,z):
        epsilon = 0.001
        a = 1.0/(1.0+np.exp(-z)+epsilon)
        return a
    def BCELoss(self,y_pred):
        l = -1*( self.y*np.log(y_pred) + (1-self.y)*np.log(1-y_pred) ) + (self.lmda/2)*np.sum(self.W*self.W)
        l/=self.n_samples
        return np.sum(l,axis=0)
    def BCELoss_dev(self,z):
        temp = z - self.y
        d = (1.0/self.n_samples)*np.matmul(np.transpose(self.X),temp) + lmda*self.W/self.n_samples
        return d
    def forward(self,X=None):
        if X is None:
            z = np.matmul(self.X,self.W)
        else:
            z = np.matmul(X,self.W)
        a = self.sigmoid(z)
        return z,a
    def train(self):
        losses = list()
        for epoch in range(self.n_epochs):
            #forward
            z,y_pred = self.forward()
            #loss
            l = self.BCELoss(y_pred)
            losses.append(l)
            #print(f'loss:{l}')
            #backprop
            w_dev = self.BCELoss_dev(z)
            #update weights
            self.W-= self.learning_rate*w_dev
        return losses
lmda = 0.01
learning_rate = 0.005
n_epochs = 500
def plot_general(x,y):
    plt.plot(x,y)
    plt.show()
#X_train.shape: (455,30)
#y_train.shape: (455,1)
#X_test.shape:  (114,30)
#y_test.shape:  (114,1)

model = LogisticRegression(X_train,y_train,n_epochs)
losses = model.train()
x = range(len(losses))
plot_general(x,losses)



#testing
m_test = X_test.shape[0]
z,y_pred = model.forward(X_test)
y_pred_rounded = np.round(y_pred)
correct = np.sum(y_pred_rounded == y_test)
print(f'accuracy:{correct/m_test}')



