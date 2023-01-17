# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing modules

import sklearn

import numpy as np

import pandas as pd

from collections import Counter

from sklearn import preprocessing

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

#loading data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head(5)

train_data.shape

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head(5)

test_data.shape

#preprocessing data



#dropping columns which are not important

train_data.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)

test_data.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)

train_data.head()

#outlier detection and removal in train data

def remove_outlier(df_in, col_name, n):

    li = []

    for col_name in col_name:



        q1 =np.percentile(df_in[col_name],25)

        q3 =np.percentile(df_in[col_name],75)

        iqr = q3-q1 #Interquartile range

        fence_low  = q1-1.5*iqr

        fence_high = q3+1.5*iqr

        df_out = df_in[(df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)].index

        li.extend(df_out)

        #print(li)

    outlier_indices = Counter(li)

    print(outlier_indices)

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )



    return multiple_outliers



df_outlier = remove_outlier(train_data, ["Age","SibSp","Parch","Fare"], 2)



train_data.loc[df_outlier]



#dropping the outlier

train = train_data.drop(df_outlier , axis = 0).reset_index(drop = True)

#converting categorical variables into indicator variables

train = pd.get_dummies(train)

test_data = pd.get_dummies(test_data)
#check for null values in train data

train.isnull().sum()

#filling null values in train data

train['Age'] = train['Age'].fillna((train['Age'].mean()))

train.head()

train.isnull().sum()

#checking null values in test data

test_data.isnull().sum()

#filling null values with mean

test_data['Age'] = test_data['Age'].fillna((test_data['Age'].mean()))

test_data['Fare'] = test_data['Fare'].fillna((test_data['Fare'].mean()))

test_data.isnull().sum()

#seperating target variables and rest of variables

X = train.drop(['Survived', 'PassengerId'], axis = 1)

Y = train['Survived'].astype(int)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.22, random_state = 0)
#applying decision tree classifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, Y_train)

survived_predict = dtc.predict(X_test)

tree_accu = metrics.accuracy_score(Y_test, survived_predict)

print(tree_accu)
#applying random forest

forest_classifier = RandomForestClassifier()

forest_classifier.fit(X_train, Y_train)

survived_predict = forest_classifier.predict(X_test)

forest_accu =  metrics.accuracy_score(Y_test, survived_predict)

print(forest_accu)
#applying linear-regression

linreg = LinearRegression()

linreg.fit(X_train, Y_train)

survived_pred = linreg.predict(X_test)

linreg_accu = metrics.accuracy_score(Y_test, survived_pred.round())

print(linreg_accu)
#applying logistic-regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

survived_pred = logreg.predict(X_test)

logreg_accu = metrics.accuracy_score(Y_test, survived_pred.round())

print(logreg_accu)
#applying support vector

svc = svm.SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

print(metrics.accuracy_score(Y_test, Y_pred.round()))

svc_accu = metrics.accuracy_score(Y_test, survived_pred.round())

#applying k-neighborsclassifier

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, Y_train)

survived_pred = model.predict(X_test)

kneighbors_accu = metrics.accuracy_score(Y_test, survived_pred.round())

print(kneighbors_accu)

#applying xgboost

import xgboost as xgb

data_dmatrix = xgb.DMatrix(data = X, label = Y)#convert data into datamatrix that xgboost support

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train,Y_train)

preds = xg_reg.predict(X_test)

xg_acc = metrics.accuracy_score(Y_test, preds.round())

print( xg_acc )

#adaboost classifier

from sklearn.ensemble import AdaBoostClassifier

#adaboost_cla = AdaBoostClassifier(n_estimators=50, learning_rate = 1)

adaboost_cla = AdaBoostClassifier(n_estimators=50,base_estimator=forest_classifier ,learning_rate = 1)

adaboost_cla.fit(X_train, Y_train)

predict = adaboost_cla.predict(X_test)

adaboost_acc = metrics.accuracy_score(Y_test, predict.round())

print( adaboost_acc)

#visualization of accuracy

import matplotlib.pyplot as plt

#%matplotlib in line

names = ['Decision tree', 'Random forest', 'Linear regression', 'Logistic regression','SVC', 'Knn classifier', 'xg boost', 'adaboost']

values = [tree_accu, forest_accu, linreg_accu, logreg_accu, svc_accu, kneighbors_accu, xg_acc, adaboost_acc]

plt.bar(names, values)

plt.tick_params(axis ='x', rotation = 90)

plt.show()

#submitting file

submission = pd.DataFrame()

submission['PassengerId'] = test_data['PassengerId']

test_data.drop(['PassengerId'], axis = 1, inplace = True)

pre = xg_reg.predict(test_data)

survived_predict = pre.round()

submission['Survived'] = survived_predict

submission.to_csv('survived_predict.csv')
