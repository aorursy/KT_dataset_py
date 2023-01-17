import numpy as np
import random
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def derivative_sigmoid(x):
    return x*(1-x)
    
def relu(x):
    return np.maximum(0,x)
    
def derivative_relu(x):
    a=x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] >0:
                a[i][j]=1
            else:
                a[i][j]=0
    return a 

def predict(w1,w2,w3,b1,b2,b3,x_test):
    z1 = np.dot(w1,x_test.T)+b1
    a1 = relu(z1)
    z2 = np.dot(w2,a1)+b2
    a2 = relu(z2)
    z3 = np.dot(w3,a2)+b3
    output = sigmoid(z3)
    return output
    
    

class MYNN:
    def __init__(self,layers,x,y,learning_rate=0.001):
        
        m = x.shape[0]
        
        self.w1 = np.random.rand(layers[1],layers[0])
        self.b1 = np.random.rand(layers[1],1)
        
        self.w2 = np.random.rand(layers[2],layers[1])
        self.b2 = np.random.rand(layers[2],1)
        
        self.w3 = np.random.rand(layers[3],layers[2])
        self.b3 = np.random.rand(layers[3],1)
        
        self.input = x.T
        self.y = y
        
        self.output = np.zeros(y.shape)
        
        self.alpha = learning_rate
        
    
    def forwardpropogation(self):
        
        self.z1 = np.dot(self.w1,self.input)+self.b1
        self.a1 = relu(self.z1)
        
        self.z2 = np.dot(self.w2,self.a1)+self.b2
        self.a2 =  relu(self.z2)
        
        self.z3 = np.dot(self.w3,self.a2)+self.b3
        self.output = sigmoid(self.z3)
    
    def backwardpropogation(self):

        self.dz3 = self.output-self.y
        self.dw3 = 1/self.y.shape[1]*np.dot(self.dz3,self.a2.T)
        self.db3 = 1/self.y.shape[1]*np.sum(self.dz3,axis=1,keepdims=True)
        
        self.dz2 = np.dot(self.dw3.T,self.dz3)*derivative_relu(self.z2)
        self.dw2 = 1/self.y.shape[1]*np.dot(self.dz2,self.a1.T)
        self.db2 = 1/self.y.shape[1]*np.sum(self.dz2,axis=1,keepdims=True)
        
        self.dz1 = np.dot(self.dw2.T,self.dz2)*derivative_relu(self.z1)
        self.dw1 = 1/self.y.shape[1]*np.dot(self.dz1,self.input.T)
        self.db1 = 1/self.y.shape[1]*np.sum(self.dz1,axis=1,keepdims=True)
        
        self.w1=self.w1-self.alpha*self.dw1
        self.b1=self.b1-self.alpha*self.db1
        
        self.w2=self.w2-self.alpha*self.dw2
        self.b2=self.b2-self.alpha*self.db2
        
        self.w3=self.w3-self.alpha*self.dw3
        self.b3=self.b3-self.alpha*self.db3
        
        return self.w1,self.b1,self.w2,self.b2,self.w3,self.b3
        
    def cost(self):
        cost=-1/self.y.shape[1]*np.sum((self.y*np.log(self.output)+(1-self.y)*np.log(1-self.output)))
        print(cost)
        return cost
layers=[3,4,5,1]
x=np.array([[0,1,2],[1,2,1],[3,4,6],[0,0,1],[1,0,1],[1,5,5],[6,1,4],[1,1,2],[2,1,1],[3,1,8]])

y=np.array([[0,0,1,0,0,1,1,0,0,1]])
print(x.shape)
nn= MYNN(layers,x,y)
cost=[]
for i in range(20000):
    nn.forwardpropogation()
    w1,b1,w2,b2,w3,b3=nn.backwardpropogation()
    cost.append(nn.cost())
print(w1,b1,w2,b2,w3,b3)
x_test=np.array([[1,0,1],[7,7,5]])
x_test.shape
print(predict(w1,w2,w3,b1,b2,b3,x_test))