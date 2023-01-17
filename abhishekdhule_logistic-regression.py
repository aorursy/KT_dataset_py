# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/marks.csv')
data.head()
data.shape
plt.figure(figsize=(12,8))
sns.scatterplot(x=data['e1_score'],y=data['e2_score'],hue=data.result)
plt.title('Exam data')
plt.show()
X=data[['e1_score','e2_score']].values   # converted into numpy array
y=data['result'].values.reshape(1,141)   # converted into numpy array and reshaped as (1,number_of_examples)
class LogisticRegression():
    def __init__(self):
        self.w=None
        self.b=None
        self.m=None
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    
    def predict(self,X):
        try:
            z=np.dot(self.w.T,X)+self.b
            y=self.sigmoid(z)
            self.m=X.shape[1]
            return y
        except:
            print('First fit the data and then predict.')
        
    def compute_loss(self,y_pred,y_true):
        cost=-(np.sum((y_true*np.log(y_pred))+(1-y_true)*np.log(1-y_pred)))/self.m
        return cost
        
    def fit(self,X,y,learning_rate=0.001,iterations=1000000):
        loss=[]
        self.w=np.zeros((X.shape[0],1))
        self.b=0
        y_pred=self.predict(X)
        loss.append(self.compute_loss(y_pred,y))
        print('loss at iteration number '+str(1)+" is: ",loss[0])
        for i in range(iterations):
            dz=(y_pred-y)
            dw=np.dot(X,dz.T)/self.m   
            db=np.sum(dz)/self.m
            self.w=self.w-learning_rate*dw
            self.b=self.b-learning_rate*db
            y_pred=self.predict(X)
            loss.append(self.compute_loss(y_pred,y))
            if (i+1)%10000==0:
                print('loss at iteration number '+str(i+1)+" is: ",loss[i])
        plt.plot(loss)
            
        
        
lr=LogisticRegression()
lr.fit(X.T,y,learning_rate=0.001,iterations=300000)
y_pred=lr.predict(X.T)
print(lr.compute_loss(y_pred,y))
l1=[]
for i in y_pred[0]:
    if i>=0.5:
        l1.append(1)
    else:
        l1.append(0)
sns.scatterplot(x=data['e1_score'],y=data['e2_score'],hue=l1)
sns.scatterplot(x=data['e1_score'],y=data['e2_score'],hue=data['result'])