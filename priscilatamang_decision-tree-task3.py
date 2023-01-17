# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data.head()
#Extracting X and Y
X=data.iloc[:,0:8].values
Y=data.iloc[:,-1].values
X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.2, random_state=1)

#Using Standard Scaler to get acuratre result
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

print("The shape of X_train:",X_train.shape)
print("The shape of X_test:",X_test.shape)
classifier=DecisionTreeClassifier()
classifier.fit(X_train,Y_train)
#Predicting
Y_predict=classifier.predict(X_test)
print(Y_predict)

#Finding Accuracy
AS=accuracy_score(Y_test,Y_predict)
print("The accuracy score:", AS)
#Creating Variable
param_dist={"criterion":["gini","entropy"],"max_depth":[1,2,3,4,5,6,7,None],"max_features":[1,2,3,4,5,6,7,None],"random_state":[0,1,2,3,4,5,6,7,8,9,None],"max_leaf_nodes":[0,1,2,3,4,5,6,7,8,9,None]}

#Applying Grid-Search-CV
grid=GridSearchCV(classifier, param_grid=param_dist, cv=10, n_jobs=-1)

#Training the model after applying Grid-Search-CV
grid.fit(X_train,Y_train)
Acc=grid.best_score_
print("The Accuracy Score is",Acc)
OHV=grid.best_params_
print("The values of Optimal Hyperparameters are",OHV)