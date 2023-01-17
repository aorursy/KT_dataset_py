# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
'''

Create train and test data



m = # of training data

t = # of testing data

'''



np.random.seed(50)

m = 100

t = 10

positive_mu = 4

positive_sd = np.array([[1,0],

                        [0,2]])



negative_mu = -1

negative_sd = np.array([[1.5,0],

                        [0,1]])



# training examples

Xtr_p = np.random.randn(m,2) @ positive_sd + positive_mu

Xtr_n = np.random.randn(int(m*.2),2) @ negative_sd + negative_mu



# testing examples

Xte_p = np.random.randn(t,2) @ positive_sd + positive_mu

Xte_n = np.random.randn(t,2) @ negative_sd + negative_mu



plt.plot(Xtr_p[:,0],Xtr_p[:,1],'.',label='train postive')

plt.plot(Xtr_n[:,0],Xtr_n[:,1],'x',label='train negative')

plt.plot(Xte_p[:,0],Xte_p[:,1],'o',label='test positive')

plt.plot(Xte_n[:,0],Xte_n[:,1],'*',label='test negative')

plt.xlabel('X1')

plt.ylabel('X2')

plt.legend()

plt.show()
class GDA:

    from scipy.stats import multivariate_normal

    

    def __init__ (self, Xtr_positive, Xtr_negative):

    

        '''

        Note:

            np.cov 

            n = # of features

            m = # of examples

            

            bias = True means devide by m, instead of m - 1

            

        input:

            Xtr_positive = (m, n)

            Xtr_negative = (m, n)

        '''

        pm = Xtr_positive.shape[0] # number of positve example

        nm = Xtr_negative.shape[0] # number of negative example

        n = Xtr_positive.shape[1]

        

        self.mu_p = np.mean(Xtr_positive, axis=0) # mu1 (n,)

        self.mu_n = np.mean(Xtr_negative, axis=0) # mu0 (n,)

        X = np.concatenate((Xtr_positive, Xtr_negative), axis=0) # combine both positive and negative examples to calculate covariance

        self.cov = np.cov(X.T, bias=True) # covarance matrix (n * n), take (n, m) matrix

        self.fi = pm /(pm + nm) # class prior, P(y=1)

        

        assert(self.cov.shape == (n, n))

        

    def predict (self, Xte):

        

        y1 = multivariate_normal.pdf(Xte, mean=self.mu_p, cov=self.cov)# P(X |y=1) 

        y0 = multivariate_normal.pdf(Xte, mean=self.mu_n, cov=self.cov)# P(X |y=0)

        

        # Bay's theorem

        p1 = (y1 * self.fi) / ((y1 * self.fi) + (y0 * (1 - self.fi)))        # P(Y = 1| X)

        p0 = (y0 * (1 - self.fi)) / ((y1 * self.fi) + (y0 * (1 - self.fi)))  # P(Y = 0| X)

        y_hat = (p1 > p0). astype(int).reshape(-1,1) # label

        

        return p1,p0, y_hat

        
model = GDA(Xtr_p, Xtr_n)

p1, p0, y_hat = model.predict(Xte_p)
p1
p0
y_hat 
x = np.linspace(-5, 5, 100, endpoint=False)

y = multivariate_normal.pdf(x, mean=0, cov=1)

plt.plot(x, y)
((Xte_p.T - np.mean(Xte_p.T, axis=1, keepdims=1)) @ (Xte_p.T - np.mean(Xte_p.T, axis=1, keepdims=1)).T) / Xte_p.shape[0]
mean = np.mean(Xte_p.T, axis=1, keepdims=True)

((Xte_p.T-mean) @ (Xte_p.T-mean).T) / Xte_p.shape[0]
cov = np.cov(Xte_p.T, bias=True)

np.diag(np.diag(cov))
import seaborn as sn

sn.heatmap(cov, annot=True)
theta = 2 * np.pi * np.arange(0,1.1,0.01)

circle = np.array([np.cos(theta), np.sin(theta)])

cov = np.cov(Xtr_p.T, bias=True)

mu_p = np.mean(Xtr_p, axis= 0, keepdims=True)

mu_n = np.mean(Xtr_n, axis= 0, keepdims=True)

std = circle * np.diag(cov).reshape(-1,1)



plt.plot(Xtr_p[:,0],Xtr_p[:,1],'.',label='train postive')

plt.plot(Xtr_n[:,0],Xtr_n[:,1],'x',label='train negative')

plt.plot(Xte_p[:,0],Xte_p[:,1],'o',label='test positive')

plt.plot(Xte_n[:,0],Xte_n[:,1],'*',label='test negative')



plt.plot(mu_p[:,0] + std[0], mu_p[:,1] + std[1])

plt.plot(mu_p[:,0] + 2*std[0], mu_p[:,1] + 2*std[1])

plt.plot(mu_p[:,0] + 3*std[0], mu_p[:,1] + 3*std[1])



plt.plot(mu_n[:,0] + std[0], mu_n[:,1] + std[1])

plt.plot(mu_n[:,0] + 2*std[0], mu_n[:,1] + 2*std[1])

plt.plot(mu_n[:,0] + 3*std[0], mu_n[:,1] + 3*std[1])



plt.xlabel('X1')

plt.ylabel('X2')

plt.legend()

plt.show()
theta = 2 * np.pi * np.arange(0,1.1,0.01)

circle = np.array([np.cos(theta), np.sin(theta)])

cov_p = np.cov(Xtr_p.T, bias=True)

cov_n = np.cov(Xtr_n.T, bias=True)

mu_p = np.mean(Xtr_p, axis= 0, keepdims=True)

mu_n = np.mean(Xtr_n, axis= 0, keepdims=True)

std_p = circle * np.diag(cov_p).reshape(-1,1)

std_n = circle * np.diag(cov_n).reshape(-1,1)



plt.plot(Xtr_p[:,0],Xtr_p[:,1],'.',label='train postive')

plt.plot(Xtr_n[:,0],Xtr_n[:,1],'x',label='train negative')

plt.plot(Xte_p[:,0],Xte_p[:,1],'o',label='test positive')

plt.plot(Xte_n[:,0],Xte_n[:,1],'*',label='test negative')



plt.plot(mu_p[:,0] + std_p[0], mu_p[:,1] + std_p[1])

plt.plot(mu_p[:,0] + 2*std_p[0], mu_p[:,1] + 2*std_p[1])

plt.plot(mu_p[:,0] + 3*std_p[0], mu_p[:,1] + 3*std_p[1])



plt.plot(mu_n[:,0] + std_n[0], mu_n[:,1] + std_n[1])

plt.plot(mu_n[:,0] + 2*std_n[0], mu_n[:,1] + 2*std_n[1])

plt.plot(mu_n[:,0] + 3*std_n[0], mu_n[:,1] + 3*std_n[1])



plt.xlabel('X1')

plt.ylabel('X2')

plt.legend()

plt.show()