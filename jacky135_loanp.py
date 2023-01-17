# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/loanprediction/train_ctrUa4K.csv')
test = pd.read_csv('/kaggle/input/loanprediction/test_lAUu6dG.csv')
train.info()
train.describe(include = 'all')
train
train.isnull().sum()
g =sns.countplot(train['Loan_Status'])
train['Loan_Status'].value_counts(normalize  = True)
sns.catplot(x = 'Property_Area',col = 'Gender',data  = train,hue= 'Loan_Status',kind = 'count')
train['Gender'].mode()[0]
train['Gender'].fillna(train['Gender'].mode()[0],inplace = True)

test['Gender'].fillna(test['Gender'].mode()[0],inplace = True)
train['Married'].fillna(train['Married'].mode()[0],inplace = True)

test['Married'].fillna(test['Married'].mode()[0],inplace = True)
train[train['Dependents'].isnull()]
sns.catplot(x = 'Dependents',col = 'Gender',data  = train,hue= 'Loan_Status',kind = 'count')
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace = True)

test['Dependents'].fillna(test['Dependents'].mode()[0],inplace = True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace = True)

test['Self_Employed'].fillna(test['Self_Employed'].mode()[0],inplace = True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean(),inplace = True)

test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mean(),inplace = True)
train['LoanAmount'].fillna(train['LoanAmount'].mode()[0],inplace = True)

test['LoanAmount'].fillna(test['LoanAmount'].mode()[0],inplace = True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace = True)

test['Credit_History'].fillna(test['Credit_History'].mode()[0],inplace = True)
train = pd.get_dummies(train,columns = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area'])

test = pd.get_dummies(test,columns = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area'])
train
y = train['Loan_Status']

train.drop(['Loan_Status','Loan_ID'],axis = 1 ,inplace = True)
id1 = test['Loan_ID']
test.drop(['Loan_ID'],axis = 1 ,inplace = True)
train
train['Credit_History'].unique()
from sklearn import preprocessing

min_max = preprocessing.MinMaxScaler().fit(train)

train_minmax  = min_max.transform(train)
train_minmax
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_minmax, y, test_size=0.33, random_state=42)
import xgboost as xgb

from sklearn.metrics import accuracy_score

xgb_clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0,

              learning_rate=0.1, max_delta_step=0, max_depth=3,

              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)

xgb_clf.fit(train_minmax,y)

y_predict = xgb_clf.predict(X_test)

accuracy = accuracy_score(y_predict,y_test)

print(accuracy)
accuracy
test_minmax  = min_max.transform(test)
y_pred = xgb_clf.predict(test_minmax)
Y_pred = pd.Series(y_pred,name="Loan_Status")

submission = pd.concat([pd.Series(id1,name="Loan_ID"),Y_pred],axis = 1)

submission.to_csv("loan.csv",index=False)



from xgboost import plot_importance

plot_importance(xgb_clf)

plt.show()
test_minmax