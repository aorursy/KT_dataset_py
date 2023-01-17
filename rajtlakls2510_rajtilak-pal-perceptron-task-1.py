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
df=pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")
df
X=df.loc[:,'CreditScore':"EstimatedSalary"]
y=df['Exited']
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
X.head()
encoded=['Geography','Gender']
for encode in encoded:
    label=LabelEncoder()
    X[encode]=label.fit_transform(X[encode])
X.head()
X[['CreditScore','Balance','EstimatedSalary']]=scaler.fit_transform(X[['CreditScore','Balance','EstimatedSalary']])
X.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
perceptron=Perceptron()
perceptron.fit(X_train,y_train)
y_pred_train=perceptron.predict(X_train)
y_pred_test=perceptron.predict(X_test)
print("Training Accuracy: ",accuracy_score(y_pred_train,y_train))
print("Testing Accuracy: ",accuracy_score(y_pred_test,y_test))
param_grid={'eta0': [1.0,0.5,1e-5], 'max_iter': [50,100]}
grid=GridSearchCV(perceptron, param_grid, cv=100)
%time grid.fit(X_train,y_train)
grid.best_score_
grid.best_params_
perceptron=grid.best_estimator_
y_pred_train=perceptron.predict(X_train)
y_pred_test=perceptron.predict(X_test)
print("Training Accuracy: ",accuracy_score(y_pred_train,y_train))
print("Testing Accuracy: ",accuracy_score(y_pred_test,y_test))
