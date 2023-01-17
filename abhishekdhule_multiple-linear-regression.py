import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
np.random.seed(20)  # for generating similar random data
def generate_data(num_features,num_examples=1000):
    x=np.random.randn(num_features,num_examples)
    w=np.random.randn(num_features,1)
    b=np.random.randn()
    y=np.dot(w.T,x)+b
    return x,y
class MultipleLinearRegression():
    def __init__(self):
        self.W=None
        self.b=None
        self.m=None
        
    def predict(self,X):
        try:
            y_pred=np.dot(self.W.T,X)+self.b
            return y_pred
        except AttributeError:
            print('First train your data using fit function and then predict')
    
    def compute_loss(self,y_pred,y_true):
        loss=np.sum(np.square(y_pred-y_true))/(2*self.m)
        return loss
    
    def fit(self,X,y,learning_rate=0.00001,iterations=1000):
        self.m=X.shape[0]
        self.W=np.random.randn(self.m,1)
        self.b=np.random.randn()
        y_pred=self.predict(X)
        loss=self.compute_loss(y_pred,y)
        losses=[]
        losses.append(loss)
        for i in range(iterations):
            dw=np.dot(X,(y_pred-y).T)/self.m
            db=np.sum(y_pred-y)/self.m
            self.W=self.W-(learning_rate*dw)
            self.b=self.b-(learning_rate*db)
            y_pred=self.predict(X)
            loss=self.compute_loss(y_pred,y)
            losses.append(loss)
            
            if (i+1)%100==0:
                print('loss at iteration number '+str(i+1)+' is : ',losses[i])
        
        plt.title('Multiple Linear Regression')
        plt.plot(losses)
        plt.xlabel('no. of iterations')
        plt.ylabel('loss')
        plt.show()
            
        
x_train,y_train=generate_data(num_features=4)
model=MultipleLinearRegression()
model.fit(x_train,y_train,iterations=2000)
y_predict=model.predict(x_train)
y_predict.shape