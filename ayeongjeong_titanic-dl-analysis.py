# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/train.csv")
train.head()
sex = {'male':0, 'female':1}
train.Sex = train.Sex.map(sex)
test.Sex = test.Sex.map(sex)
train.Sex.isnull().sum()
train.Embarked.unique()
embarked = {'S':0, 'C':1, 'Q':2}
train.Embarked  = train.Embarked.map(embarked)
test.Embarked  = test.Embarked.map(embarked)
train.isnull().sum()
test.isnull().sum()
sns.countplot('Embarked', data=train)
train.Embarked = train.Embarked.fillna(0)
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

train['Title'] = train['Title'].map(title_mapping)
test['Title'] = test['Title'].map(title_mapping)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
y_train = train['Survived']
x_train = train.drop(['PassengerId','Cabin','Survived','Name','Ticket'],axis=1)

y_test = test['Survived']
x_test = test.drop(['PassengerId','Cabin','Survived','Name','Ticket'],axis=1)
model = Sequential()
model.add(Dense(10, input_shape=(8,),activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50)
model.evaluate(x_test, y_test)
