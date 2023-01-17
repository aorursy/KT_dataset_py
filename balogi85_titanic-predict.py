# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
import datetime
import tensorflow as tf
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = '/kaggle/input/titanic/train.csv'
test = '/kaggle/input/titanic/test.csv'
train_data = pd.read_csv(train)
test_data = pd.read_csv(test)
train_data.drop('PassengerId',inplace=True, axis=1)
test_data.drop('PassengerId',inplace=True, axis=1)
def cab_replace(data):
  a = data['Cabin'].str[0]
  b = data['Cabin'].str[1:]
  str(a)
  a.replace('A',1, inplace=True)
  a.replace('B',2, inplace=True)
  a.replace('C',3, inplace=True)
  a.replace('D',4, inplace=True)
  a.replace('E',5, inplace=True)
  a.replace('F',6, inplace=True)
  a.replace('G',7, inplace=True)
  a.replace('T',8, inplace=True)
  a.fillna(0,inplace=True)
  b.fillna(0,inplace=True)
  data['Cabin_a']=a
  data['Cabin_b']=b
  data.drop('Cabin',axis=1,inplace=True)
  data['Sex'].replace('male',1, inplace=True)
  data['Sex'].replace('female',0,inplace=True)
  data['Embarked'].replace('S',1,inplace=True)
  data['Embarked'].replace('C', 2, inplace=True)
  data['Embarked'].replace('Q', 3, inplace=True)
  data['Age'].fillna(data['Age'].median(), inplace=True)
  data['Embarked'].fillna(data['Embarked'].median(),inplace=True)
  
def round_2(data):
  data.drop('Ticket',axis=1,inplace=True)
  data['title'] = data['Name'].map(lambda x: re.search(', (.+?) ', x).group(1))
  title_array = data['title'].unique()
  a = 1
  for i in data['title']:
    for j in title_array:
        if i == j: data['title'].replace(i,a,inplace=True)
    a+=1
  data.drop('Name',axis=1,inplace=True)
  data.drop('Cabin_b',axis=1,inplace=True)
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epochs, logs={}):
    if logs.get('accuracy') > 0.894:
         print('\nCallback is stopped the training!')
         model.stop_training=True
cab_replace(train_data)
cab_replace(test_data)
round_2(train_data)
round_2(test_data)
y_train = train_data['Survived']
train_data.drop('Survived',axis=1,inplace=True)
max_col = list(train_data.max())
x_train = train_data.astype('float32')
x_test = test_data.astype('float32')
x_train.head()
x_test.head()
print(x_train.shape,x_test.shape)
callback = myCallback()
model = Sequential()
model.add(Dense(200, activation='relu', input_dim = x_train.shape[1]))
model.add(BatchNormalization())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=20, epochs=400, verbose=2, callbacks = [callback])
Prediction = model.predict(x_test)
test_id = pd.read_csv(test)
y_predict = (Prediction > 0.5).astype(int).reshape(x_test.shape[0])
print(len(y_predict))
out = pd.DataFrame({'PassengerId': test_id['PassengerId'], 'Survived': y_predict})
out.to_csv('pr.csv', index=False)
