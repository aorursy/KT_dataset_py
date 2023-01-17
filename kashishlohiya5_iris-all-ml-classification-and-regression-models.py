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
#importing and reading dataset iris
data=pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
print(data.head(10))
data.columns
X=data.iloc[:,:4]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
#Logistic Regression
from sklearn.linear_model import LogisticRegression

model_log=LogisticRegression()
model_log.fit(X_train,y_train)
#save all scores of different models
tot_score= []
pred=model_log.predict(X_test)
score_log=model_log.score(X_test,y_test)*100
tot_score.append(score_log)
print(score_log)
#K Nearest Neighbour Model

from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier()
model_knn.fit(X_train,y_train)

pred=model_knn.predict(X_test)
score_knn=model_knn.score(X_test,y_test)*100
tot_score.append(score_knn)
print(score_knn)
#SupportVectorMachine Model
from sklearn.svm import SVC

model_svc = SVC()
model_svc.fit(X_train, y_train)

pred = model_svc.predict(X_test)
score_svc=model_svc.score(X_test,y_test)*100
tot_score.append(score_svc)
print(score_svc)
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

pred_nb = model_nb.predict(X_test)
score_nb=model_nb.score(X_test,y_test)*100
tot_score.append(score_nb)
print(score_nb)
#Decision Tree model

from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier()

model_dt.fit(X_train, y_train)

pred_dt = model_dt.predict(X_test)

score_dt=model_dt.score(X_test,y_test)*100
tot_score.append(score_dt)
print(score_dt)
#Linear Regression

from sklearn.linear_model import LinearRegression
dummy_data=pd.get_dummies(data.species)
dummy_data.head()
mdata=pd.concat([data,dummy_data],axis='columns')
mdata.head()
fdata=mdata.drop(['species'],axis='columns')
fdata.head()
X=fdata.iloc[:,:5]
y=fdata.iloc[:,5:8]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
#creating object of linear regression

model_linreg=LinearRegression()
model_linreg.fit(X_train,y_train)
pred_lr=model_linreg.predict(X_test)
score_lr=model_linreg.score(X_test,y_test)*100
print(score_lr)
tot_score.append(score_lr)
from sklearn.metrics import mean_squared_error 
mse = mean_squared_error(y_test, pred_lr)

print("Mean Square Error : ", mse) 
#print('Mean Absolute Error:', model_linreg.mean_absolute_error(y_test, pred_lr))
#print('Mean Squared Error:', mean_squared_error(y_test, pred_lr))
#print('Mean Root Squared Error:', np.sqrt(mean_squared_error(y_test, pred_lr)))
tot_score