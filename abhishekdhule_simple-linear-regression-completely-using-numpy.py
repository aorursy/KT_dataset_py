#Simple Linear Regression

#Importing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
#generating data for simple linear regression
np.random.seed(1)

x=np.random.randn(1000,1)  
w=np.random.randn()
b=np.random.randn()

y=x*w+b +(np.random.randn(1000,1)*0.09)  #this last term of np.random.randn() is to add noise in data

# plt.title('Generated data')
plt.figure(figsize=(12,8))
plt.title('Generated data')
plt.xlabel('input(x)')
plt.ylabel('output(y)')
sns.scatterplot(x=x[:,0],y=y[:,0])
plt.grid(True)
plt.show()

class SimpleLinearRegression():
    def __init__(self):
        self.parameters={}
        self.m=None
    
    
    def predict(self,X):
        self.m=X.shape[0]
        try:
            y_pred=self.parameters['w1']*X+self.parameters['b']
            return y_pred
        except KeyError:
            print('First fit your data and then predict the values')

            
    def cost(self,y_pred,y_true):
        return (np.sum(np.square(y_pred-y_true)))/(2*self.m)
     
        
    def fit(self,X,y,learning_rate=0.01,iterations=1000):
        self.parameters['w1']=np.random.randn()
        self.parameters['b']=np.random.randn()
        cost=[]
        y_pred=self.predict(X)
        cost.append(self.cost(y_pred,y))
        print('Cost at iteration number '+str(1)+' is : ',cost[0])
        for i in range(iterations):
            dw1=np.sum((y_pred-y)*X)/self.m
            db=np.sum(y_pred-y)/self.m
            
            self.parameters['w1']=self.parameters['w1']-(learning_rate*dw1)
            self.parameters['b']=self.parameters['b']-(learning_rate*db)
            
            y_pred=self.predict(X)
            cost.append(self.cost(y_pred,y))
            if (i+1)%100==0:
                print('Cost at iteration number '+str(i+1)+' is : ',cost[i])
        plt.title('cost v/s iteration')
        plt.plot(cost)
        plt.xlabel('no. of iterations')
        plt.ylabel('cost')
        plt.show()
model=SimpleLinearRegression()
model.fit(x,y)
y_pred=model.predict(x)
cost=model.cost(y_pred,y)
print(cost)
plt.figure(figsize=(12,8))
plt.title('Simple Linear Regression')
sns.scatterplot(x=x[:,0],y=y[:,0],alpha=0.4)
sns.lineplot(x=x[:,0],y=y_pred[:,0])
plt.xlabel('x-input')
plt.ylabel('y-input')
plt.show()