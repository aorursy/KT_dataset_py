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
df_train = pd.read_csv('../input/train.csv')
train = df_train.copy()
train.head(20)
train.info()
train['Sex'] = train['Sex'].map({'male':0 , 'female':1})
train['Age'] = train['Age'].fillna(round(train['Age'].mean()))
train['Embarked'] = train['Embarked'].fillna(value = 'S')
train['Embarked'] = train['Embarked'].map({'S':0, 'C':1, 'Q':2})
train.head()
train.columns.values
train = train[['PassengerId','Pclass', 'Name', 'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
train.head()
unscaled_inputs = train.iloc[:,:-1]
unscaled_inputs
from sklearn.preprocessing import StandardScaler
titanic_scaler = StandardScaler()
titanic_scaler.fit(unscaled_inputs)
scaled_inputs = titanic_scaler.transform(unscaled_inputs)
scaled_inputs.shape
targets = train['Survived']
targets.sum()/targets.shape[0]
scaled_inputs
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
reg = LogisticRegression()
reg.fit(scaled_inputs, targets)
model_outputs = reg.predict(scaled_inputs)

model_outputs
targets.values
model_outputs == targets.values
sum(model_outputs == targets.values)
sum(model_outputs == targets.values)/model_outputs.shape[0]
reg.coef_
reg.intercept_[0]
features_name = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
summary_table = pd.DataFrame(columns = ['Features'], data = features_name)
summary_table['Coef'] = np.transpose(reg.coef_)
summary_table['Coef']
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept',reg.intercept_[0]]
summary_table
summary_table.sort_index()
summary_table['Odds Ratio'] = np.exp(summary_table['Coef'])
summary_table
summary_table.sort_values('Odds Ratio',ascending = False)
df_test = pd.read_csv('../input/test.csv')
test = df_test.copy()
test.info()
test['Sex'] = test['Sex'].map({'male':0 , 'female':1})
test['Fare'] = test['Fare'].fillna(value = test['Fare'].mean())
test['Age'] = test['Age'].fillna(round(test['Age'].mean()))
test['Embarked'] = test['Embarked'].fillna(value = 'S')
test['Embarked'] = test['Embarked'].map({'S':0, 'C':1, 'Q':2})
test = test[['PassengerId','Pclass', 'Name', 'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
unscaled_test_inputs = test.iloc[:,:]
titanic_scaler.fit(unscaled_test_inputs)
scaled_test_inputs = titanic_scaler.transform(unscaled_test_inputs)
unscaled_test_inputs.shape
prediction = reg.predict(scaled_test_inputs)
prediction.sum()/scaled_test_inputs.shape[0]
prediction.shape
scaled_test_inputs.shape
unscaled_test_inputs.shape
data = pd.read_csv('../input/test.csv')
data['PassengerId'].head()
survival = pd.DataFrame(columns = ['PassengerId'], data = data['PassengerId'])
survival['Survived'] = np.transpose(prediction)
survival.head()
survival.to_csv('survival.csv')