import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()
raw_data = pd.read_csv('/kaggle/input/titanic/train.csv')

raw_data.head()
raw_data.describe()
data = raw_data.dropna(how='any', subset=['Age', 'Embarked'])
data.nunique()
data = data.drop_duplicates(subset=['Ticket'])
data.shape
target = data['Survived']
data = data.drop(['Name', 'Cabin', 'Ticket'], axis = 1)
data.head()
data.isnull().sum()
data['Sex'] = data['Sex'].map({'male':0,'female':1})
data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2})
data.head()
corr = data.corr()

corr.style.background_gradient()
data.shape
inputs = data.drop(['Survived'], axis=1)
target.shape
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()
scaler.fit_transform(inputs)
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.2, random_state=10)
model.fit(x_train, y_train)
model.predict(x_test)
model.score(x_test, y_test)