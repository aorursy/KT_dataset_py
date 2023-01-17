# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

X, y= make_regression(n_samples=100, n_features=1, noise=0.4, bias=50)
class LassoReg():
    def __init__(self,lr,itr,l1):
        self.lr=lr
        self.itr=itr
        self.l1=l1
    def fit(self,X,y):
        self.m,self.n=X.shape
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.y=y
        
        for _ in range(self.itr):
            self.update_weights()
        return self
    def update_weights(self):
        y_pred=self.predict(self.X)
        dw=np.zeros(self.n)
        for j in range(self.n):
            if self.w[j]>0:
                dw[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.y - y_pred ) )  
                           
                         + self.l1 ) / self.m 
            else:
                dw[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.y - y_pred ) )  
                           
                         - self.l1 ) / self.m 
        db = - 2 * np.sum( self.y - y_pred ) / self.m  
        self.w = self.w-self.lr*dw
        self.b = self.b-self.lr*db
        return self
     
    def predict( self, X ) : 
      
        return X.dot( self.w ) + self.b 
      
        
mylr = LassoReg(0.05,1000,50)
mylr.fit(X,y)
mylr.predict(X)
y
