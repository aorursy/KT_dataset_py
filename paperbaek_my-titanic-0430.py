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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.describe()
train['Pclass'] = train['Pclass'].astype(str)
train.isnull().sum()
train = train[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Survived']]
train_drop = train.dropna()
x = train_drop[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = train_drop['Survived']
x_train = x.copy()
x_train['Age'] = (x['Age'] - x['Age'].mean() ) / x['Age'].std()
x_train['Fare'] = (x['Fare'] - x['Fare'].mean() ) / x['Fare'].std()
x_train.describe()
x_encoded = pd.get_dummies(x_train[['Pclass', 'Sex', 'Embarked']])
x_encoded
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
model = Sequential([
    Dense(10, activation='relu', input_shape=(8,)),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['acc'])
model.fit(x_encoded, y, epochs=30, validation_split=0.1)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.info()
test['Pclass'] = test['Pclass'].astype(str)
test = test[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
test
test_fill = test.fillna(test.mean())
x = test_fill[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
x_test = x.copy()
x_test['Age'] = (x['Age'] - x['Age'].mean() ) / x['Age'].std()
x_test['Fare'] = (x['Fare'] - x['Fare'].mean() ) / x['Fare'].std()
x_test_encoded = pd.get_dummies(x_test[['Pclass', 'Sex', 'Embarked']])
x_test_encoded
answer = model.predict(x_test_encoded)
answer.shape
sub = list()
for a in answer:
    if a >= 0.5:
        sub.append(1)
    else:
        sub.append(0)
np.mean(sub)
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission
submission['Survived'] = sub
submission
submission.to_csv('submission.csv', index=False)