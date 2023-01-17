# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data['Sex_Indicator']=train_data['Sex'].map({'male':1,'female':0}).astype(int)

embark_null = train_data['Embarked'].mode()[0]

train_data['Embark_Indicator'] = (train_data['Embarked'].fillna(embark_null)

                                  .map({'S':0,'C':1,'Q':2}).astype(int))

test_data['Sex_Indicator']=test_data['Sex'].map({'male':1,'female':0}).astype(int)

embark_null = test_data['Embarked'].mode()[0]

test_data['Embark_Indicator'] = (test_data['Embarked'].fillna(embark_null)

                                  .map({'S':0,'C':1,'Q':2}).astype(int))
for i in range(0, 2):

        for j in range(0, 3):

            train_data.loc[(train_data['Age'].isnull())&(train_data['Sex_Indicator'] == i)

                           & (train_data['Pclass'] == j+1),'Age'] = train_data[(train_data['Sex_Indicator'] == i) 

                                                                                & (train_data['Pclass'] == j+1)]['Age'].dropna().median()

            train_data.loc[(train_data['Fare'].isnull())&(train_data['Sex_Indicator'] == i)

                           & (train_data['Pclass'] == j+1),'Fare'] = train_data[(train_data['Sex_Indicator'] == i) 

                                                                                & (train_data['Pclass'] == j+1)]['Fare'].dropna().median()

            test_data.loc[(test_data['Age'].isnull())&(test_data['Sex_Indicator'] == i)

                           & (test_data['Pclass'] == j+1),'Age'] = test_data[(test_data['Sex_Indicator'] == i) 

                                                                                & (test_data['Pclass'] == j+1)]['Age'].dropna().median()

            test_data.loc[(test_data['Fare'].isnull())&(test_data['Sex_Indicator'] == i)

                           & (test_data['Pclass'] == j+1),'Fare'] = test_data[(test_data['Sex_Indicator'] == i) 

                                                                                & (test_data['Pclass'] == j+1)]['Fare'].dropna().median()
train_data['With_Family'] = [0 if(Parch == 0 & Sib ==0) else 1 for Parch,Sib in zip(train_data['Parch'],train_data['SibSp'])]

train_data['Family_Size'] = train_data['Parch']+train_data['SibSp']+1

test_data['With_Family'] = [0 if(Parch == 0 & Sib ==0) else 1 for Parch,Sib in zip(test_data['Parch'],test_data['SibSp'])]

test_data['Family_Size'] = test_data['Parch']+test_data['SibSp']+1
train_data['Suffix'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.')

train_data['Suffix'] = train_data['Suffix'].replace('Mlle', 'Miss')

train_data['Suffix'] = train_data['Suffix'].replace('Ms', 'Miss')

train_data['Suffix'] = train_data['Suffix'].replace('Mme', 'Mrs')

train_data['Suffix'] = train_data['Suffix'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Outlier')
test_data['Suffix'] = test_data['Name'].str.extract(' ([A-Za-z.]+)\.')

test_data['Suffix'] = test_data['Suffix'].replace('Mlle', 'Miss')

test_data['Suffix'] = test_data['Suffix'].replace('Ms', 'Miss')

test_data['Suffix'] = test_data['Suffix'].replace('Mme', 'Mrs')

test_data['Suffix'] = test_data['Suffix'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Outlier')
pd.crosstab(train_data['Suffix'],train_data['Sex'])
train_data['Suffix_Indicator'] = train_data['Suffix'].map({'Master':1,'Miss':2,'Mr':3,'Mrs':4,'Outlier':5})

test_data['Suffix_Indicator'] = test_data['Suffix'].map({'Master':1,'Miss':2,'Mr':3,'Mrs':4,'Outlier':5})
test_data.head()
#Train Data Preparation

from sklearn.model_selection import train_test_split

# Took the following features to determine the survival rate

X_train,X_test,y_train,y_test = (train_test_split

                                 (train_data[['Pclass','Sex_Indicator','Age','SibSp',

                                              'Parch','Embark_Indicator','Fare','With_Family','Family_Size','Suffix_Indicator']],train_data['Survived'],random_state=0))
from sklearn.preprocessing import MinMaxScaler

Scaler = MinMaxScaler()

X_train_scaler = Scaler.fit_transform(X_train)

X_test_scaler = Scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report,roc_auc_score,precision_score,recall_score

LoR = LogisticRegression(random_state=0)

c_values = {'C':[1,15,10,100,150,250]}

grdClf = GridSearchCV(LoR,param_grid=c_values,scoring='precision')

grdClf.fit(X_train_scaler,y_train)

y_decs = grdClf.decision_function(X_test_scaler)

(grdClf.best_params_,grdClf.best_score_,roc_auc_score(y_test,y_decs))
from sklearn.model_selection import cross_val_score

cross_val_score(DTClf,X_train,y_train,cv=5,scoring='precision')
# Logistic Regression Model with Regularization param set to 10 as per GridSearchCV best_params_

LR = LogisticRegression(C=10,random_state=0).fit(X_train_scaler,y_train)

(LR.score(X_train_scaler,y_train),LR.score(X_test_scaler,y_test))
from sklearn.ensemble import RandomForestClassifier

rndClf = RandomForestClassifier(n_estimators=100,random_state=0)

rndClf.fit(X_train_scaler,y_train)

(rndClf.score(X_train_scaler,y_train),rndClf.score(X_test_scaler,y_test))
from sklearn.tree import DecisionTreeClassifier

DTClf = DecisionTreeClassifier()

DTClf.fit(X_train_scaler,y_train)

(DTClf.score(X_train_scaler,y_train),DTClf.score(X_test_scaler,y_test))
#Probablity of Titanic Survival Passenger List

y_prob = rndClf.predict(Scaler.transform(test_data[['Pclass','Sex_Indicator','Age','SibSp',

                                              'Parch','Embark_Indicator','Fare','With_Family','Family_Size','Suffix_Indicator']]))

test_data['Survived'] = y_prob
test_data[['PassengerId','Survived']].to_csv('test_predictions.csv',sep=',',encoding='utf-8',index=False)
df = pd.read_csv('test_predictions.csv')
df