# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/pga9-classification-hackathon/hackathon_train.csv')

test = pd.read_csv('/kaggle/input/pga9-classification-hackathon/hackathon_test.csv')

sample_submission = pd.read_csv('/kaggle/input/pga9-classification-hackathon/sample_submission.csv')
train.head()
test.head()
train.info()
sns.heatmap(train.isnull())
sns.pairplot(train,hue = 'default.payment.next.month',vars = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE'])
sns.countplot(train['default.payment.next.month'])
train.columns
sns.scatterplot(x = 'BILL_AMT1',y= 'BILL_AMT2',hue = 'default.payment.next.month',data=train)
plt.figure(figsize=(20,15))

sns.heatmap(train.corr(),annot = True)
train['SEX'].value_counts()
# train.drop('ID',inplace = True,axis = 1)

# test.drop('ID',inplace = True,axis = 1)
X = train.drop('default.payment.next.month',axis = 1)
y = train['default.payment.next.month']
from sklearn.model_selection import train_test_split
X_train,X_test , y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)
y_pred = svc_model.predict(X_test)
from sklearn.metrics import classification_report , confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
print(cr)
sns.heatmap(cm,annot = True,fmt='d')
#use the normalization to increase the accuracy

min_train = X_train.min()

print(min_train)
range_train = (X_train - min_train).max()

print(range_train)
X_train_scaled = (X_train - min_train)/range_train
X_train_scaled.head()
min_test = X_test.min()

range_test = (X_test - min_test).max()

X_test_scaled = (X_test - min_test)/range_test
svc_model.fit(X_train_scaled,y_train)
y_pred = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot = True,fmt = 'd')
cr = classification_report(y_test,y_pred)
print(cr)
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred = logistic_regression.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot = True,fmt = 'd')
cr = classification_report(y_test,y_pred)
print(cr)
logistic_regression.fit(X_train_scaled,y_train)
param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV



grid = GridSearchCV(SVC(),param_grid=param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)
grid.best_params_
grid_prediction = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test,grid_prediction)
sns.heatmap(cm,annot = True,fmt = 'd')
cr = classification_report(y_test,grid_prediction)
print(cr)