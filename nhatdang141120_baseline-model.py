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
#metric

from sklearn.metrics import mean_absolute_error
#read data

train_data=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test_data=pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

sample_submission=pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')

print(train_data.head())

print(test_data.head())

print(sample_submission.head())

print(train_data.columns)

print('Train data shape',train_data.shape)
#check columns with categorical feature and check NaN

num_cols=train_data._get_numeric_data().columns



print(num_cols)

print(set(train_data.columns)-set(num_cols))

print(train_data.columns[train_data.isna().any()].tolist())
#set up trainning data and test data

from sklearn.impute import KNNImputer

y = train_data.SalePrice

X=train_data.dropna(axis=1)._get_numeric_data().drop(['SalePrice','Id'],axis=1)

X_test=test_data[X.columns].dropna(axis=1)

X=X[X_test.columns]

imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

imputer.fit(X_test)

imputer.fit_transform(X_test)

X_test=X_test.fillna(X_test.mean())

#split the data

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
#test different model

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

## Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(train_X, train_y)

y_pred = gaussian.predict(val_X)

print('Gaussian Naive Bayes: ',mean_absolute_error(y_pred,val_y))

#Logistic Regression

lg=LogisticRegression().fit(train_X, train_y)

y_pred2=lg.predict(val_X)

print('Logistic Regression: ',mean_absolute_error(y_pred2,val_y))

# Support Vector Machines

from sklearn.svm import SVC

svc=SVC().fit(train_X,train_y)

y_pred3=svc.predict(val_X)

print('Support vector machine: ',mean_absolute_error(y_pred3,val_y))
#linear svc

from sklearn.svm import LinearSVC

lnsvc=LinearSVC().fit(train_X,train_y)

y_pred4=lnsvc.predict(val_X)

print('linear svc: ',mean_absolute_error(y_pred4,val_y))
# Perceptron

from sklearn.linear_model import Perceptron

pct=Perceptron().fit(train_X,train_y)

y_pred5=pct.predict(val_X)

print('perceptron: ',mean_absolute_error(y_pred5,val_y))
# Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier().fit(train_X,train_y)

y_pred6=rfc.predict(val_X)

print('Random Forest Classifier',mean_absolute_error(y_pred6,val_y))
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier().fit(train_X,train_y)

y_pred7=dtc.predict(val_X)

print('Decision Tree Classifier',mean_absolute_error(y_pred7,val_y))

# KNN or k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3).fit(train_X,train_y)

y_pred8=knn.predict(val_X)

print('KNeighborsClassifier',mean_absolute_error(y_pred8,val_y))
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

SGD=SGDClassifier().fit(train_X,train_y)

y_pred9=SGD.predict(val_X)

print('Stochastic Gradient Descent',mean_absolute_error(y_pred9,val_y))
#output

prediction=rfc.predict(X_test)

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': prediction})

output.to_csv('submission.csv', index=False)



