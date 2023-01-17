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
#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

from scipy import stats

from statsmodels.stats.outliers_influence import variance_inflation_factor

%matplotlib inline





GenderSubmission = pd.read_csv("/kaggle/input/gender_submission.csv")

Test = pd.read_csv("/kaggle/input/test.csv")

Train = pd.read_csv("/kaggle/input/train.csv")



#Describe Data

print(Train.describe())



#Check for null/nan values 

print((Train.isnull().sum()/(Train.shape[0]))*100)



#Create Seperate Variable for Testing

Traind=Train



#Replace NaNs

Traind["Age"].fillna(Traind["Age"].median(skipna=True), inplace=True)

Traind["Embarked"].fillna(Traind['Embarked'].value_counts().idxmax(), inplace=True)

#Dummy Generation

Traind['Single'] = (Traind['SibSp']+Traind['Parch'])==0

Traind = pd.get_dummies(Train,columns=['Pclass', 'Sex', 'Embarked','Single'])

print(Traind.head())



#Create Training Variables/Model

X = Traind.drop(columns=['Name','PassengerId','Ticket','Cabin','SibSp','Parch','Survived','Single_False','Pclass_2','Embarked_C','Sex_female'])

y = Traind['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LogisticRegression()

lm.fit(X_train,y_train)



#Evaluate Model



#I removed FamilySize_1 due to p-score and high multicollinearity

print('Coefficients: \n', lm.coef_)

predictions = lm.predict( X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.round(1)



#Now using Test Data

#Create Seperate Variable for Testing

Testd=Test

#Replace NaNs

Testd["Age"].fillna(Testd["Age"].median(skipna=True), inplace=True)

Testd["Embarked"].fillna(Testd['Embarked'].value_counts().idxmax(), inplace=True)

#Dummy Generation

Testd['Single'] = (Testd['SibSp']+Testd['Parch'])==0

Testd = pd.get_dummies(Test,columns=['Pclass', 'Sex', 'Embarked','Single'])

#Get Fare Averages to replace NaN

Testd[Testd["Fare"].isnull()]=Testd[Testd['Pclass_1'] == 0][Testd['Pclass_2'] == 0].mean()[4]



#Submission Creation

Testing = Testd.drop(columns=['Name','PassengerId','Ticket','Cabin','SibSp','Parch','Single_False','Pclass_2','Embarked_C','Sex_female'])

Testing['Survived']=lm.predict(Testing)

Testing['PassengerId'] = Test['PassengerId']

submission = Testing[['PassengerId', 'Survived']]

submission.to_csv("submission.csv", index=False)