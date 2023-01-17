# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('/kaggle/input/titanic/train.csv')
dataset.head(5)
dataframe = pd.DataFrame(dataset)
def missing_percentage(df):     
    missing_total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([missing_total, percent], axis=1, keys=['Missing_Total','Percent'])


missing_percentage(dataframe)
dataframe.drop(['Cabin'], axis = 1, inplace = True)
dataframe.head(5)
#Fill the missing values in Age
dataframe['Initial']=0
for i in dataframe:
    dataframe['Initial']=dataframe.Name.str.extract('([A-Za-z]+)\.')

dataframe['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',
                         'Jonkheer','Col','Rev','Capt','Sir','Don'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other',
                         'Other','Other','Mr','Mr','Mr'],inplace=True)

dataframe.groupby('Initial')['Age'].mean()
#Fill with Values
dataframe.loc[(dataframe.Age.isnull())&(dataframe.Initial=='Mr'),'Age']=33
dataframe.loc[(dataframe.Age.isnull())&(dataframe.Initial=='Mrs'),'Age']=36
dataframe.loc[(dataframe.Age.isnull())&(dataframe.Initial=='Master'),'Age']=5
dataframe.loc[(dataframe.Age.isnull())&(dataframe.Initial=='Miss'),'Age']=22
dataframe.loc[(dataframe.Age.isnull())&(dataframe.Initial=='Other'),'Age']=46
#Delete observation without Embark
#dataframe.drop(dataframe[pd.isnull(dataframe['Embarked'])].index, inplace = True)
dataframe['Embarked'] = dataframe['Embarked'].fillna('S')
#Checking the null value
missing_percentage(dataframe)
#just Keep the original copy for furture use
dataframe_cy = dataframe.copy()
dataframe.drop(['Name'], axis = 1, inplace = True)
dataframe.drop(['Ticket'], axis = 1, inplace = True)
dataframe.drop(['Initial'], axis = 1, inplace = True)
dataframe.drop(['PassengerId'], axis = 1, inplace = True)
dataframe.head(5)
dataframe = pd.get_dummies(dataframe, drop_first=True)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
dataframe['Age'] = scalar.fit_transform(dataframe[['Age']])
dataframe['Fare'] = scalar.fit_transform(dataframe[['Fare']])
dataframe['Pclass'] = scalar.fit_transform(dataframe[['Pclass']])
dataframe.head(5)
y = dataframe[['Survived']]
x = dataframe.drop(['Survived'], axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(x, y, test_size = 0.3, random_state = 1234)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_predict = lr.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_predict)
cm
lr.score(X_test, Y_test)
from sklearn.svm import SVC
svc = SVC()

svc.fit(X_train, Y_train)
Y_predict_svc = svc.predict(X_test)

#Lets Check Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict_svc)
cm
svc.score(X_test, Y_test)
svc_rbf = SVC(kernel='rbf', gamma = 1)
svc_rbf.fit(X_train, Y_train)
Y_predict_svc1 = svc_rbf.predict(X_test)

cm_rbf = confusion_matrix(Y_test, Y_predict_svc1)
cm_rbf
print('Accuracy_RBF ', svc_rbf.score(X_test, Y_test))
svc_lin = SVC(kernel='linear')
svc_lin.fit(X_train, Y_train)
Y_predict_svc2 = svc_lin.predict(X_test)

cm_lin = confusion_matrix(Y_test, Y_predict_svc2)
cm_lin
print('Accuracy_Lin ', svc_lin.score(X_test, Y_test))
svc_poly = SVC(kernel='poly')
svc_poly.fit(X_train, Y_train)
Y_predict_svc3 = svc_poly.predict(X_test)

cm_poly = confusion_matrix(Y_test, Y_predict_svc3)
cm_poly
print('Accuracy_POLY ', svc_lin.score(X_test, Y_test))
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1234)
dtc.fit(X_train, Y_train)

Y_predict_dtc = dtc.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm_dtc = confusion_matrix(Y_test, Y_predict_dtc)
cm_dtc
print('DecisionTree ', dtc.score(X_test, Y_test))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)
rfc.fit(X_train, Y_train)

Y_predict_rfc = rfc.predict(X_test)

#confusion matrix
cm_rfc = confusion_matrix(Y_test, Y_predict_rfc)
cm_rfc
print('RandomForest ', rfc.score(X_test, Y_test))
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_predict_svc))
dataframe_test = pd.read_csv('/kaggle/input/titanic/test.csv')
dataframe_test.head(5)
missing_percentage(dataframe_test)
#Fill the missing values in Age
dataframe_test['Initial']=0
for i in dataframe_test:
    dataframe_test['Initial']=dataframe_test.Name.str.extract('([A-Za-z]+)\.')

dataframe_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',
                         'Jonkheer','Col','Rev','Capt','Sir','Don'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other',
                         'Other','Other','Mr','Mr','Mr'],inplace=True)

dataframe_test.groupby('Initial')['Age'].mean()
dataframe_test.loc[(dataframe_test.Age.isnull())&(dataframe_test.Initial=='Mr'),'Age']=33
dataframe_test.loc[(dataframe_test.Age.isnull())&(dataframe_test.Initial=='Mrs'),'Age']=36
dataframe_test.loc[(dataframe_test.Age.isnull())&(dataframe_test.Initial=='Master'),'Age']=5
dataframe_test.loc[(dataframe_test.Age.isnull())&(dataframe_test.Initial=='Miss'),'Age']=22
dataframe_test.loc[(dataframe_test.Age.isnull())&(dataframe_test.Initial=='Other'),'Age']=46
#dataframe_test.drop(dataframe_test[pd.isnull(dataframe_test['Fare'])].index, inplace = True)
dataframe_test['Fare'].fillna(dataframe_test['Fare'].median(), inplace=True)
missing_percentage(dataframe_test)
dataframe_test.drop(['Cabin'], axis = 1, inplace = True)
missing_percentage(dataframe_test)
dataframe_test.head(5)
dataframe_test.drop(['Name'], axis = 1, inplace = True)
dataframe_test.drop(['Ticket'], axis = 1, inplace = True)
dataframe_test.drop(['Initial'], axis = 1, inplace = True)
dataframe_test.head(5)
dataframe_test = pd.get_dummies(dataframe_test, drop_first=True)
dataframe_test.shape
scalar_test = StandardScaler()
scalar_test.fit_transform(dataframe_test)
dataframe_test.head(5)
dataframe_cy.drop(['Name'], axis = 1, inplace = True)
dataframe_cy.drop(['Ticket'], axis = 1, inplace = True)
dataframe_cy.drop(['Initial'], axis = 1, inplace = True)
#dataframe_cy.drop(['Survived'], axis = 1, inplace = True)
dataframe_cy = pd.get_dummies(dataframe_cy, drop_first=True)
dataframe_cy.head(5)
svm_test_data = SVC()
svm_test_data.fit(dataframe_cy.drop(['Survived'], axis = 1), dataframe_cy['Survived'])
test_predict = svm_test_data.predict(dataframe_test)
test_predict.shape
print('SVC_TEST ', svm_test_data.score(dataframe_cy.drop(['Survived'], axis = 1),
                                 dataframe_cy['Survived']))
test_pred = pd.DataFrame(test_predict, columns = ['Survived'])
test_data_new = pd.concat([dataframe_test, test_pred], axis = 1, join = 'inner')
df_sub = test_data_new[['PassengerId', 'Survived']]
df_sub.head()
df_sub.to_csv('prediction.csv', index = False)
