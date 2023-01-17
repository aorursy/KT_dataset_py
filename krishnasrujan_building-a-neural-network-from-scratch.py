class MY_Neural_Network:
    def __init__(self,layers,x,y,learning_rate=0.001):
        
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
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)
    
class MY_Neural_Network:
    def __init__(self,layers,x,y,learning_rate=0.001):
        
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
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def derivative_sigmoid(x):
    return x*(1-x)
    
def relu(x):
    return np.maximum(0,x)
    
def derivative_relu(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] >0:
                x[i][j]=1
            else:
                x[i][j]=0
    return x

class MY_Neural_Network:
    def __init__(self,layers,x,y,learning_rate=0.001):
        
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
        self.a2 = relu(self.z2)
        
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
        return cost