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
import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test_raw = pd.read_csv('../input/test.csv')
y_train = train['Survived']

x_train = train.drop('Survived', axis=1)
dataset = pd.concat([x_train, test_raw])
dataset.info()
dataset.head()
total = dataset.isnull().sum().sort_values(ascending=False)

percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()
dataset = dataset.drop('Cabin', axis=1)
dataset.Embarked.fillna('N/A',inplace=True)
dataset.Age.fillna(dataset.Age.mean(), inplace=True)
dataset.Fare.fillna(dataset.Fare.mean(), inplace=True)
dataset
len(x_train)
dataset = dataset.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
from sklearn.preprocessing import LabelEncoder

for column in ['Sex', 'Embarked']:

    dataset[column] = LabelEncoder().fit_transform(dataset[column])
dataset
train_len = len(x_train)

x_train = dataset[:train_len]

test = dataset[train_len:]
import tensorflow as tf

import keras

from keras import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(50, input_dim=7, init='uniform', activation='relu'))

model.add(Dense(25, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.3)
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
result = model.predict(test)

result = pd.Series([int(round(x[0])) for x in result], name='Survived')
submission = test_raw['PassengerId']

submission = pd.concat([submission, result], axis=1)
submission.to_csv('submission.csv', index=False)