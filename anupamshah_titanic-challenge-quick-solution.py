# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

sub_data = pd.read_csv('../input/titanic/gender_submission.csv')
train_data.head()
train_data.shape
print(train_data.isnull().sum())
print(test_data.isnull().sum())
test_data.shape
train_data = train_data.drop(['Cabin'], axis = 1)

test_data = test_data.drop(['Cabin'], axis = 1)

train_data = train_data.drop(['Ticket'], axis = 1)

test_data = test_data.drop(['Ticket'], axis = 1)
train_data.head()
train_data.dropna(axis=1)

test_data.dropna(axis=1)
train_data.drop([ "Name" ,"Embarked"], inplace = True, axis = 1 )
train_data.head()
train_data['SP'] = train_data['SibSp'] + train_data['Parch']
train_data.head()
train_data.isnull().sum()
train_data['Age'].fillna((train_data['Age'].median()), inplace=True)
sex_mapping = {"male":0, "female":1}

train_data['Sex'] = train_data['Sex'].map(sex_mapping) 
train_data.head()
train_data.info()
y_train = train_data['Survived']

x_train = train_data.drop(['Survived'], axis = 1)
test_data.head()
test_data.drop([ "Name" ,"Embarked"], inplace = True, axis = 1 )

test_data['SP'] = test_data['SibSp'] + test_data['Parch']

test_data['Sex'] = test_data['Sex'].map(sex_mapping) 
test_data.head()
test_data.isnull().sum()
test_data['Age'].fillna((test_data['Age'].median()), inplace=True)

test_data['Fare'].fillna((test_data['Fare'].mean()), inplace=True)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42,n_estimators=100,max_depth=5)



fit = model.fit(x_train,y_train)
ids = test_data['PassengerId']

test_data = test_data.dropna(axis=1)

predictions = model.predict(test_data)
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)