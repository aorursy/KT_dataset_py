
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.info()
test.isnull().sum()
train.drop('Education', axis=1, inplace=True)
test.drop('Education', axis=1, inplace=True)
train.head(30)
train.drop('Self_Employed', axis=1, inplace=True)
test.drop('Self_Employed', axis=1, inplace=True)
train.head(30)
train.Gender=pd.get_dummies(train.Gender)
train.Married=pd.get_dummies(train.Married)
train.head(30)
train=pd.get_dummies(train,columns=['Property_Area'])
train.head(30)
test.Gender=pd.get_dummies(test.Gender)
test.Married=pd.get_dummies(test.Married)
test=pd.get_dummies(test,columns=['Property_Area'])
test.head()
train.isnull().sum()
test.isnull().sum()
train.head()
train.drop('Property_Area_Rural', axis=1, inplace=True)
test.drop('Property_Area_Rural', axis=1, inplace=True)
train["LoanAmount"].fillna(train["LoanAmount"].mean(), inplace=True)
train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mean(), inplace=True)
train["Credit_History"].fillna(train["Credit_History"].mean(), inplace=True)

train.head()
train.isnull().sum()
train['Dependents'].value_counts()
train.Dependents=train.Dependents.replace(["3+"],["4"])
train['Dependents'].value_counts()
train.isnull().sum()
train["Dependents"].fillna(train["Dependents"].median(), inplace=True)
train.isnull().sum()
test.head(30)
test["LoanAmount"].fillna(test["LoanAmount"].mean(), inplace=True)
test["Loan_Amount_Term"].fillna(test["Loan_Amount_Term"].mean(), inplace=True)
test["Credit_History"].fillna(test["Credit_History"].mean(), inplace=True)
test['Dependents'].value_counts()
test.Dependents=test.Dependents.replace(["3+"],["4"])
test["Dependents"].fillna(test["Dependents"].median(), inplace=True)
test.isnull().sum()
test.head()
train.head()
train.drop('Loan_ID', axis=1, inplace=True)
test.drop('Loan_ID', axis=1, inplace=True)
    
target = train['Loan_Status']
train.drop('Loan_Status', axis=1, inplace=True)

train.shape, target.shape
train.head()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 10)
scoring = 'accuracy'
score = cross_val_score(clf, train, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
import numpy as np
round(np.mean(score)*100, 2)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100, 2)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100, 2)
#so basically i am partially correct using 