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
import pandas as pd
import numpy as np
from numpy import log, dot, e
from numpy.random import rand
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
X = load_breast_cancer()['data']
y = load_breast_cancer()['target']
feature_names = load_breast_cancer()['feature_names']
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 227
plt.rcParams['figure.figsize'] = (16,3)
plt.barh(['Malignant','Benign'],[sum(y), len(y)-sum(y)], height=0.3)
plt.title('Class Distribution', fontSize=15)
plt.show()
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

class myLogisticRegression:
    
    def sigmoid(self,z):
        return 1/(1+e**(-z))
    
    def cost_function(self,X,y,weights):
        z=dot(X,weights)
        pred_0 = y * log(self.sigmoid(z))
        pred_1 = (1 - y) * log(1 - self.sigmoid(z)) 
        return -sum(pred_1 + pred_0) / len(X)
    def fit(self,X,y,epochs=100,lr=0.05):
        loss=[]
        weights=rand(X.shape[1])
        N=len(X)
        
        for _ in range(0,epochs):
            y_hat = self.sigmoid(dot(X,weights))
            weights -= lr*dot(X.T,y_hat-y)/N
            loss.append(self.cost_function(X,y,weights))
        
        self.weights = weights
        self.loss = loss
        
    def predict(self,X):
        z=dot(X,self.weights)
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]
logreg = myLogisticRegression()
logreg.fit(X_train, y_train, epochs=500, lr=0.5)
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))
print('-'*55)
print('Confusion Matrix\n')
print(confusion_matrix(y_test, y_pred))
