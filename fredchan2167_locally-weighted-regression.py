# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
X = np.arange(0,10,0.1).reshape(-1,1)
y = np.sin(X)
X += np.random.normal(scale=.1,size=(100,1))

plt.plot(X,y,'.')
plt.title('Data')
plt.show()
def polynomial_X (X,degree=1):
    '''
    map feature vecotr X to polynomial
    
    input:
    X: feature matrix with shape (m,n) w/o X0=1
    degree: level of degree wish to map to
    
    return:
    degreeX: ploynomial X with size = (m, n*degree)
    '''
    
    m = X.shape[0]
    n = X.shape[1]
    
    degreeX = np.zeros((m,n*degree))
    
    index = 0
    for i in range(degree):
        for j in range(n):
            degreeX[:,index] = X[:,j]**(i+1)
            index += 1
        
    return degreeX
    


def normalize(X):
    '''
    Normalize data with mean = 0 and standard deviation = 1
    
    input:
    X: feature matrix with size (m,n), m = # example, n = # of feature
    
    return:
    X: with mean = 0 and sd = 1
    '''
    
#     m = X.shape[0]
#     mu = np.mean(X, axis=0, keepdims=True)
#     sd = np.std(X, axis=0, keepdims=True)
#     X = (X - mu) / sd
        
    return X

def concat_ones(X):
    '''
    concatenate X0 = 1 to X for bias
    '''
    
    ones = np.ones((X.shape[0],1))
    X = np.concatenate((ones,X), axis=1)
    return X

class linear_regression:
    
    def __init__ (self,Xtr, Y):
        self.Xtr = Xtr
        self.Y = Y

        
    def train (self,iteration = 1000, lr = 0.01,degree=1):
        costs=[]
        self.d = degree
        
        self.Xtr = polynomial_X(self.Xtr,degree = self.d)
        self.Xtr = normalize(self.Xtr)
        self.Xtr = concat_ones(self.Xtr)
        
        
        m = self.Xtr.shape[0]
        n = self.Xtr.shape[1]
        
        self.theta = np.zeros((n,1))

        for i in range(iteration):
            h = self.Xtr.dot(self.theta)

            J = (1/(2*m)) * (h - self.Y).T @ (h - self.Y) 
            dJ = self.Xtr.T @ (h - self.Y)

            self.theta = self.theta - lr * dJ
            if i % 100 == 0:
                costs.append(J)
            
        return self.theta
    
    def predict(self,Xte):
        Xte = polynomial_X(Xte, degree = self.d)
        Xte = normalize(Xte)
        Xte = concat_ones(Xte)

        return Xte @ self.theta
    
    def normal_equation (self):
        '''
        theta = inverse(X'X)X'Y
        '''
        self.normal = np.linalg.inv(self.Xtr.T@self.Xtr)@self.Xtr.T@self.Y
        return self.normal
model = linear_regression(X,y)
theta = model.train(iteration=1500, lr= 0.0003)
normal = model.normal_equation()
theta
normal
plt.plot(X,y,'.')
plt.plot([0,max(X)],[normal[0],max(X)*normal[1]+normal[0]],'-',color='r')
plt.show()
def concat_ones(X):
    '''
    concatenate X0 = 1 to X for bias
    '''
    
    ones = np.ones((X.shape[0],1))
    X = np.concatenate((ones,X), axis=1)
    return X

def locally_regression(X,X_hat,Y,tau=0.1):
    '''
    m = # of examples
    n = # of features
    
    input: 
            X = training data with shape (m, n)
            X_hat = test data with shape (1, n)
            Y = Traing label with shape (m, 1)
            tau = bandwith paramenter, standard deviation of gassian distribution
                  larger tau means include more local data for prediction, underfitting
                  smaller tau means overfitting
                  
    return:
            prediction of X_hat
            
    Pro:
        Can fit curvy data without expanding to polynomial model
    Con:
        training and predicting in the same step thus model size grows linear with amount of data
        with large n, inverse become expensive
    
    Use locally regression when features are relatively low and  large training data (<10k)
        
    '''
    
    # add bias
    X = concat_ones(X) # (m, n+1)
    X_hat = np.r_[1,X_hat]# (1, n+1)
    
    #Gassuan Kernel
    W = (np.exp((np.sum(X-X_hat,axis=1)**2) / (-2*tau*tau))).reshape(-1,1) # (m, 1)

    
    XW = X * W # shape (m, n), weighted
    theta = np.linalg.inv(XW.T @ X) @ XW.T @Y 
    theta = theta.reshape(-1,1) # (n+1, 1)
   
    
    return theta.T @X_hat
    
    
    

#prediction
locally_regression(X,6,y,tau=0.2)
#ground truth
np.sin(6)
def plot_lwr(X,Y,tau):
    
    domain = np.arange(0,10,0.1)
    prediction = [locally_regression(X,X_hat,Y,tau) for X_hat in domain]
    
    
    plt.plot(X,Y,'.')
    plt.title('tau=%g' %tau)
    plt.plot(domain,prediction,'-',color='r')
    plt.show()
plot_lwr(X,y,0.05)
plot_lwr(X,y,0.1)
plot_lwr(X,y,1)
plot_lwr(X,y,5)
