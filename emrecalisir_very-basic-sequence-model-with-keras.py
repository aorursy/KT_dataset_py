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
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.models import load_model
df = pd.read_csv("../input/train.csv")
df.shape
# replacing NaN values with the mean of column values
df = df.fillna(df.mean())
df[['Age']]=df[['Age']].astype(int)
df['Age_child'] = np.where(df['Age']<=15, 1, 0)
df['Age_others'] = np.where(df['Age']>15, 1, 0)
one_hot_gender = pd.get_dummies(df['Sex'])
df = df.drop('Sex', axis=1)
df = df.join(one_hot_gender)

one_hot_embarked = pd.get_dummies(df['Embarked'])
df = df.drop('Embarked', axis=1)
df = df.join(one_hot_embarked)
predictors=df[['Pclass','female','male','C','Q','S','SibSp','Parch','Age_child','Age_others']]
n_cols = predictors.shape[1]
input_shape = (n_cols,)
target = to_categorical(df.Survived)
model = Sequential()
# Add the first layer
model.add(Dense(32,activation='relu',input_shape=input_shape))

# Add more layers 
# for i in range (1,4):
#  model.add(Dense(100,activation='relu'))

model.add(Dense(2,activation='softmax'))
#learning_rate_to_test = [0.000001, 0.01, 1]
#for lr in learning_rate_to_test:
#    print('\n\nTesting model with learning rate: %f\n'%lr )
#    my_optimizer = SGD(lr=lr)
#    model.compile(optimizer=my_optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
#    model.fit(predictors, target,validation_split=0.3, epochs=30)

# the best learning rate is 1
my_optimizer = SGD(lr=1)
model.compile(optimizer=my_optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
#early_stopping_monitor = EarlyStopping(patience=1)
#model.fit(predictors, target,validation_split=0.3, epochs=100,callbacks=[early_stopping_monitor])
model.fit(predictors, target,validation_split=0.3, epochs=50)
df_test = pd.read_csv("../input/test.csv")
df_test.shape
df_test = df_test.fillna(df_test.mean())
df_test[['Age']]=df_test[['Age']].astype(int)
df_test['Age_child'] = np.where(df_test['Age']<=15, 1, 0)
df_test['Age_others'] = np.where(df_test['Age']>15, 1, 0)

one_hot_gender = pd.get_dummies(df_test['Sex'])
df_test = df_test.drop('Sex', axis=1)
df_test = df_test.join(one_hot_gender)

one_hot_embarked = pd.get_dummies(df_test['Embarked'])
df_test = df_test.drop('Embarked', axis=1)
df_test = df_test.join(one_hot_embarked)

predictors_test=df_test[['Pclass','female','male','C','Q','S','SibSp','Parch','Age_child','Age_others']]
predictions = model.predict(predictors_test)
res = pd.Series(np.argmax(predictions, axis=1))
df_test['Survived']=res
df_test_result = df_test[['PassengerId','Survived']]
# This model had 0.77 score on the test set
