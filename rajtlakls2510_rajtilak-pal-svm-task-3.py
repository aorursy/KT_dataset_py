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
df=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
X=df.iloc[:,2:-1]
y=df['diagnosis']
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train.shape
y_train.shape
svm=SVC()
svm.fit(X_train,y_train)
y_pred_train=svm.predict(X_train)
y_pred_test=svm.predict(X_test)
print("Training Accuracy: ",accuracy_score(y_train,y_pred_train))
print("Testing Accuracy: ",accuracy_score(y_test,y_pred_test))

svm=SVC()
param_grid=[]
param_grid.append({'kernel': ['linear'], 'C': [0.01,0.1,1,10,100,200,500,1000]})
param_grid.append({'kernel': ['poly'], 'C': [0.01,0.1,1,10,100,200,500,1000], 'degree': range(2,11)})
param_grid.append({'kernel': ['rbf'], 'C': [0.01,0.1,1,10,100,200,500,1000]})
param_grid
gd=GridSearchCV(svm,param_grid,cv=10,n_jobs=-1,scoring='accuracy')
gd.fit(X_train,y_train)
gd.best_params_
gd.best_estimator_
gd.best_score_
svm=gd.best_estimator_
y_pred_train=svm.predict(X_train)
y_pred_test=svm.predict(X_test)
print("Training Accuracy: ",accuracy_score(y_train,y_pred_train))
print("Testing Accuracy: ",accuracy_score(y_test,y_pred_test))
