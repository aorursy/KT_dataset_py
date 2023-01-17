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
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/train.csv')

#processing training set
X = dataset.copy()
y = dataset.iloc[:, 1]
X['Age'].fillna(X['Age'].median(),inplace=True)
X['Embarked'].fillna('S',inplace=True)
X.drop(['PassengerId','Ticket','Cabin'],axis = 1,inplace=True)
X.drop('Name',axis=1,inplace=True)
X.drop('Survived',axis=1,inplace=True)
#Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X['Sex'] = labelencoder.fit_transform(X['Sex'])
X['Embarked'] = labelencoder.fit_transform(X['Embarked'])
X['Embarked'] = pd.get_dummies(X['Embarked'])

#processing test set
test = pd.read_csv('../input/test.csv')
Z = test.copy()
Z['Age'].fillna(Z['Age'].median(),inplace=True)
Z['Embarked'].fillna('S',inplace=True)
Z.drop(['PassengerId','Ticket','Cabin','Name'],axis = 1,inplace=True)
Z['Fare'].fillna(Z['Fare'].median(),inplace=True)
Z['Sex'] = labelencoder.fit_transform(Z['Sex'])
Z['Embarked'] = labelencoder.fit_transform(Z['Embarked'])
Z['Embarked'] = pd.get_dummies(Z['Embarked'])

Z.isnull().sum()

data = pd.read_csv('../input/gender_submission.csv')

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

y_pred = regressor.predict(Z)

data['y_p'] = y_pred

data['y_p'] = data['y_p'].apply(lambda x:0 if x<0.5 else 1)

data['k'] = (data['Survived'] == data['y_p'])
print(data['k'].value_counts())
regressor.score(Z,y_pred)
submission = pd.DataFrame({
        "PassengerId": data["PassengerId"],
        "Survived": data['y_p']
    })
