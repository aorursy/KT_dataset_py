# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()
df.shape
#cleaning of data: that not requred

df.drop('EmployeeNumber', axis = 1, inplace = True)

df.drop('Over18', axis = 1, inplace = True)

df.drop('StandardHours', axis = 1, inplace = True)

df.drop('EmployeeCount', axis =1, inplace = True)
df.columns
df.shape
df.head(2)
y = df['Attrition']

X = df.drop('Attrition', axis = 1)
y.unique()
y = pd.get_dummies(y, drop_first = True)
df.info()
df.select_dtypes(['object'])
ind_BusinessTravel = pd.get_dummies(df['BusinessTravel'], prefix = 'BusinessTravel', drop_first = True)

ind_Department = pd.get_dummies(df['Department'], prefix = 'Department', drop_first = True)

ind_EducationField = pd.get_dummies(df['EducationField'], prefix = 'EducationField', drop_first = True)

ind_Gender = pd.get_dummies(df['Gender'], prefix = 'Gender', drop_first = True)

ind_JobRole = pd.get_dummies(df['JobRole'], prefix = 'JobRole', drop_first = True)

ind_MaritalStatus = pd.get_dummies(df['MaritalStatus'], prefix = 'MaritalStatus', drop_first = True)

ind_OverTime = pd.get_dummies(df['OverTime'], prefix = 'OverTime', drop_first = True)
ind_BusinessTravel.head()
df['BusinessTravel'].unique()
df.select_dtypes(['int64']).head(2)
sns.heatmap(df.isnull())
df1 = pd.concat([ind_BusinessTravel, ind_Department, ind_EducationField, ind_Gender, 

                 ind_JobRole, ind_MaritalStatus, ind_OverTime, df.select_dtypes(['int64'])], axis=1)
df1.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df1,y, test_size = 0.3,random_state= 42)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state = 42)

clf.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score,cross_val_predict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def print_score(clf, X_train, y_train, X_test, y_test, train = True):

    '''

    print the accuracy score, classification report and confusion matrix of classifier

    '''

    if train:

        '''

        training performance

        '''

        print("Train Result:\n")

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))

        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))



        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')

        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))

        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

        

    elif train==False:

        '''

        test performance

        '''

        print("Test Result:\n")        

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))

        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))    

        

print_score(clf, X_train, y_train, X_test, y_test,train=True)
print_score(clf, X_train, y_train, X_test, y_test, train = False)
from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(base_estimator = clf, n_estimators = 5000,

                           bootstrap = True, n_jobs = -1, random_state = 42 )
bag_clf.fit(X_train, y_train)
print_score (bag_clf, X_train, y_train, X_test, y_test, train = True )
print_score(bag_clf, X_train, y_train, X_test, y_test, train = False)
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()

rf_clf.fit(X_train, y_train.values.ravel())
print_score(rf_clf, X_train, y_train.values.ravel(),X_test, y_test.values.ravel(), train = True),
print_score(rf_clf, X_train, y_train.values.ravel(),X_test, y_test.values.ravel(), train = False),
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier()

ada_clf.fit(X_train, y_train.values.ravel())

print_score(ada_clf, X_train, y_train.values.ravel(), X_test, y_test.values.ravel(), train = True),
print_score(ada_clf, X_train, y_train.values.ravel(), X_test, y_test.values.ravel(), train = False)
ada_clf = AdaBoostClassifier(RandomForestClassifier())

ada_clf.fit(X_train, y_train.values.ravel())

print_score(ada_clf, X_train, y_train.values.ravel(), X_test, y_test.values.ravel(), train = True)
from sklearn.ensemble import GradientBoostingClassifier
gbc_clf = GradientBoostingClassifier()

gbc_clf.fit(X_train, y_train.values.ravel())
print_score(gbc_clf, X_train, y_train.values.ravel(), X_test, y_test.values.ravel(),train = True)

print_score(gbc_clf, X_train, y_train.values.ravel(), X_test, y_test.values.ravel(), train = False )
import xgboost as xgb
xgb_clf = xgb.XGBClassifier()

xgb_clf.fit(X_train, y_train.values.ravel())
print_score(xgb_clf, X_train, y_train.values.ravel(), X_test, y_test.values.ravel(), train = True )
print_score(xgb_clf, X_train, y_train.values.ravel(), X_test, y_test.values.ravel(), train = False )