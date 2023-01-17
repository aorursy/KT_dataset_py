# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
gender = pd.read_csv('../input/gender_submission.csv')
          
print('train:')
print(train_df.describe())
print('test:')
print(test_df.describe())
print(test_df.shape)
print('gender submition:')
print(gender.describe())
print('')

#null values on Age and Cabin
print('null values count:')
print(train_df.isnull().sum())

#I need to categorize the Sex to 1(male) or 0(female)
train_df['C_sex'] = pd.Categorical(train_df['Sex'], train_df['Sex'].unique()).codes
test_df['C_sex'] = pd.Categorical(test_df['Sex'], test_df['Sex'].unique()).codes
from sklearn.model_selection import train_test_split
#I chose to fill those NaN values for Age instead of dropping them
# train_df = train_df.fillna(method='ffill')
train_df = train_df.dropna()
train_x, test_x, train_y, test_y = train_test_split(train_df[['C_sex', 'Age', 'Fare']], train_df['Survived'], test_size=0.3, random_state=42)
train_x = train_x.values
train_y = train_y.values.T
test_x = test_x.values
test_y = test_y.values.T

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
import keras as k
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.models import Model, Sequential

# create model
model = Sequential()
model.add(Dense(4, input_dim=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

validate_x = test_df[['C_sex', 'Age', 'Fare']].values
Y = gender['Survived'].values

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=train_x, y=train_y, epochs=15000)

score = model.evaluate(x=test_x, y=test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
raw_predictions = model.predict(x=validate_x)

aux = pd.read_csv('../input/test.csv')

predictions_df = pd.DataFrame(raw_predictions)
predictions_df['out'] = predictions_df.mean(axis=1)
predictions_df['PassengerId'] = aux['PassengerId']
predictions_df['out'] = predictions_df['out'].map(lambda s: 1 if s >= 0.5 else 0)

predictions_df = predictions_df[['PassengerId', 'out']]
predictions_df.columns = ['PassengerId', 'Survived']
predictions_df.to_csv('submission_3Layers_no_Drop.csv', index=False)
