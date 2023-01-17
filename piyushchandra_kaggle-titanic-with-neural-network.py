# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head(10)
test.head(10)
x_train=train.drop("Survived",axis=1)

y_train=train['Survived']
x_train=x_train.drop('PassengerId',axis=1)

x_train=x_train.drop('Name',axis=1)

x_train=x_train.drop('Ticket',axis=1)

x_train=x_train.drop('Cabin',axis=1)
x_test=test

x_test=x_test.drop('Name',axis=1)

x_test=x_test.drop('PassengerId',axis=1)

x_test=x_test.drop('Ticket',axis=1)

x_test=x_test.drop('Cabin',axis=1)
train.plot(kind='scatter',x='Survived',y='Age')

plt.show()
train.plot(kind='scatter',x='Survived',y='Pclass')

plt.show()
train.plot(kind='scatter',x='Survived',y='Fare')

plt.show()
x_train=pd.get_dummies(x_train)

x_test=pd.get_dummies(x_test)
x_train.info()
x_test.info()
x_train.Age.fillna(x_train.Age.mean(), inplace=True)

x_test.Age.fillna(x_test.Age.mean(), inplace=True)

x_test.Fare.fillna(x_test.Fare.mean(),inplace=True)
train_stats=x_train.describe()

train_stats=train_stats.transpose()

train_stats
test_stats=x_test.describe()

test_stats=test_stats.transpose()

test_stats
def norm_train(x):

  return (x - train_stats['mean']) / train_stats['std']

x_train=norm_train(x_train)
def norm_test(x):

  return (x - test_stats['mean']) / test_stats['std']

x_test=norm_test(x_test)
model=tf.keras.Sequential([

    tf.keras.layers.Dense(10,activation='relu',input_shape=[len(x_train.keys())]),

    tf.keras.layers.Dense(20,activation='relu'),

    tf.keras.layers.Dense(8,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])

model.compile(optimizer='adam',loss='binary_crossentropy',

             metrics=['accuracy'])

model.summary()
model.fit(x_train,y_train,epochs=15,validation_split=0.1)
loss,acc=model.evaluate(x_train,y_train)
pred=model.predict(x_test)

pred=np.around(pred)

prediction = pd.DataFrame(pred, columns=['Survived']).to_csv('y_titanic.csv')