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
np.random.seed(50)
n = 50

positive_mu = 4
positive_sd = np.array([[1,0],
                        [0,2]])

negative_mu = 0
negative_sd = np.array([[1.5,0],
                        [0,1]])

positive = np.random.randn(n,2) @ positive_sd + positive_mu
negative = np.random.randn(n,2) @ negative_sd + negative_mu

test_positive = np.random.randn(10,2) @ positive_sd + positive_mu
test_negative = np.random.randn(10,2) @ negative_sd + negative_mu
plt.plot(positive[:,0],positive[:,1],'.',label='positive')
plt.plot(negative[:,0],negative[:,1],'x',label='negative')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
plt.plot(positive[:,0],positive[:,1],'.',label='positive')
plt.plot(negative[:,0],negative[:,1],'x',label='negative')
plt.plot(test_positive[:,0],test_positive[:,1],'o',label='test positive')
plt.plot(test_negative[:,0],test_negative[:,1],'*',label='tes negative')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
Y_positive = np.ones((n,1))
Y_negative = np.zeros((n,1))
Y = np.concatenate((Y_positive,Y_negative),axis=0)
X = np.concatenate((positive,negative), axis=0)
data = np.concatenate((X,Y), axis=1)
np.random.shuffle(data)
Xtr = data[:,:2]
Ytr = data[:,-1].reshape(-1,1)
Y_positive = np.ones((10,1))
Y_negative = np.zeros((10,1))
Y = np.concatenate((Y_positive,Y_negative),axis=0)
X = np.concatenate((test_positive,test_negative), axis=0)
data = np.concatenate((X,Y), axis=1)
np.random.shuffle(data)
Xte = data[:,:2]
Yte = data[:,-1].reshape(-1,1)
def sigmoid (z):
    return 1/(1 + np.exp(-z))

def decision_boundary (X,theta):
    '''
    input:
        X = np array (m, 2)
        theta = np array (3, 1)
        
    return:
        plot_X = minimum / maximum X cordinate 
        plot_Y = minimum / maximum Y cordinate 
    '''
    minX , maxX = min(X[:,0]) , max(X[:,0])
    plot_Y = (theta[1] * minX + theta[0]) / -theta[2] , (theta[1] * maxX + theta[0]) / -theta[2]
    plot_X = (minX, maxX)
    
    return plot_X , plot_Y

class logistic_regression:
    
    
    def __init__ (self,Xtr,Ytr):
        '''
        input:
            Xtr = (m, n)
            Ytr = (m, 1)
        '''
        self.Ytr = Ytr
        self.Xtr = Xtr
        ones = np.ones((self.Xtr.shape[0],1))
        self.Xtr = np.concatenate((ones,self.Xtr), axis=1)
        
    
    def train(self,iteration = 1000, lr = 0.01):
        '''
        Gradient Accent
        Cross Entropy Loss
        
        '''
        costs = []
        
        m , n = self.Xtr.shape
        
        self.theta = np.zeros((n,1))
        
        for i in range(iteration):
            h = sigmoid(self.Xtr @ self.theta)
            
            cost = (1/m)* np.sum(self.Ytr * np.log(h) + (1 - self.Ytr) * np.log(1 - h))
            grad = self.Xtr.T @ (self.Ytr - h)
            self.theta = self.theta + lr * grad
            
            if i % 100 == 0:
                costs.append(cost)
            
        return costs, self.theta
    
    def newton (self,iteration = 30):
        '''
        h = sigmoid(X @ theta)
        gradient = X.T @ (h - y)
        S = diag(h * (1 - h))
        H = X.T @ S @ X
        theta = theta + inv(H) @ gradient
        
        Pro:
            almost always require less iteration than gradient decent
            quadratic convergence, error: 0.01 -> 0.0001 -> 0.00000001
            Great for # of features < 50
        Con:
            inverse Hassian can be expensive for # of features > 50
        
        '''
        costs = []
        
        m , n = self.Xtr.shape
        
        self.new_theta = np.zeros((n,1))
        
        for i in range(iteration):
            h = sigmoid(self.Xtr @ self.new_theta)
            
            cost = (1/m)* np.sum(self.Ytr * np.log(h) + (1 - self.Ytr) * np.log(1 - h))
            grad = self.Xtr.T @ (self.Ytr - h)
            S = np.diagflat(h * (1 - h))
            H = self.Xtr.T @ S @ self.Xtr
            self.new_theta = self.new_theta + np.linalg.pinv(H) @ grad
            
            if i % 1 == 0:
                costs.append(cost)
            
        return costs, self.new_theta
        
    def predict (self, Xte, Yte):
        
        ones = np.ones((Xte.shape[0], 1))
        y_hat = ((np.concatenate((ones,Xte), axis=1) @ self.theta) > 0.5).astype(int)
        return np.mean(y_hat == Yte), y_hat
    
    def new_predict (self, Xte, Yte):
        
        ones = np.ones((Xte.shape[0], 1))
        y_hat = ((np.concatenate((ones,Xte), axis=1) @ self.new_theta) > 0.5).astype(int)
        return np.mean(y_hat == Yte), y_hat
        
model = logistic_regression(Xtr,Ytr)
costs, theta = model.train(iteration = 4000, lr = 0.0007)
new_costs, new_theta = model.newton(iteration = 5)
matrix, y_hat = model.predict(Xte,Yte)
new_matrix, new_y_hat = model.new_predict(Xte,Yte)
print(f'Gradient Decent Cost: {costs[-1]:.5f}, Accuracy: {matrix}')
print(f'Newton Method Cost: {new_costs[-1]:.5f}, Accuracy: {new_matrix}')
plt.plot(costs,'-', label='Gradient Accent')
plt.plot(new_costs,'-', label='Newton')
plt.title('Costs')
plt.xlabel('epoch')
plt.legend()
plt.show()
plot_X, plot_Y = decision_boundary(Xtr, theta)
new_plot_X, new_plot_Y = decision_boundary(Xtr, new_theta)

plt.plot(positive[:,0],positive[:,1],'.',label='positive')
plt.plot(negative[:,0],negative[:,1],'x',label='negative')
plt.plot(plot_X,plot_Y, '-',label='Gradient Accent')
plt.plot(new_plot_X,new_plot_Y, '-',label='Newton')
plt.xlabel('X0')
plt.ylabel('X1')
plt.legend()
plt.show()

