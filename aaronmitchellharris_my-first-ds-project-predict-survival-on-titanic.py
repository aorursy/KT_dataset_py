# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
raw_train = pd.read_csv('../input/train.csv')

raw_test = pd.read_csv('../input/test.csv')

train = raw_train.copy()

test = raw_test.copy()

train.head()
train['Sex'] = train['Sex'].map({'male':0,'female':1})

test['Sex'] = test['Sex'].map({'male':0,'female':1})



train['Embarked'] = train['Embarked'].map({'S':0,'C':1,'Q':2})

test['Embarked'] = test['Embarked'].map({'S':0,'C':1,'Q':2})
train.describe()
print(train.info(), test.info())
train.loc[pd.isnull(train['Embarked']),'Embarked'] = train['Embarked'].median()

train.loc[pd.isnull(train['Age']),'Age'] = train['Age'].median()



test.loc[pd.isnull(test['Fare']), ['Fare']] = test['Fare'].median()

test.loc[pd.isnull(test['Age']), ['Age']] = train['Age'].median()
train['Fare'] = preprocessing.scale(train['Fare'])

train['SibSp'] = preprocessing.scale(train['SibSp'])

train['Parch'] = preprocessing.scale(train['Parch'])

train['Age'] = preprocessing.scale(train['Age'])



test['Fare'] = preprocessing.scale(test['Fare'])

test['SibSp'] = preprocessing.scale(test['SibSp'])

test['Parch'] = preprocessing.scale(test['Parch'])

test['Age'] = preprocessing.scale(test['Age'])
# get dummy variables

pclass_dum = pd.get_dummies(train['Pclass'])

sex_dum = pd.get_dummies(train['Sex'])

embarked_dum = pd.get_dummies(train['Embarked'])



# rename the columns

pclass_dum.columns = ('Pclass 1', 'Pclass 2', 'Pclass 3')

sex_dum.columns = ('Male', 'Female')

embarked_dum.columns = ('S', 'C', 'Q')



# drop the original columns from training data

train = train.drop(columns=['Pclass','Sex','Embarked','Name','Ticket','Cabin'])



# add the new dummy variables into our training data

train = train.join(pclass_dum)

train = train.join(sex_dum)

train = train.join(embarked_dum)



# get dummy variables

pclass_dum = pd.get_dummies(test['Pclass'])

sex_dum = pd.get_dummies(test['Sex'])

embarked_dum = pd.get_dummies(test['Embarked'])



# rename the columns

pclass_dum.columns = ('Pclass 1', 'Pclass 2', 'Pclass 3')

sex_dum.columns = ('Male', 'Female')

embarked_dum.columns = ('S', 'C', 'Q')



# drop the original columns from testing data

test = test.drop(columns=['Pclass','Sex','Embarked','Name','Ticket','Cabin'])



# add the new dummy variables into our testing data

test = test.join(pclass_dum)

test = test.join(sex_dum)

test = test.join(embarked_dum)



train.head()
x_train = train.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13]]

y_train = train.iloc[:, [1]]



log_reg = LogisticRegression().fit(x_train, y_train)
x_test = test.drop(columns=['PassengerId'])



predict_test =  log_reg.predict(x_test)
prediction = pd.DataFrame({

    "PassengerId": test['PassengerId'],

    "Survived": predict_test

})



prediction = prediction.astype(int)

prediction
prediction.to_csv(path_or_buf='prediction.csv',index=False)