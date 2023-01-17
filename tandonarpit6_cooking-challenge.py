# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.io.json import json_normalize

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
training_data=pd.read_json('../input/train.json')
test_data=pd.read_json('../input/test.json')
training_data.head()
ingredients=[]
cuisines=[]
for ingredient in training_data['ingredients']:
    for element in ingredient:
        if element not in ingredients:
            ingredients.append(element)
len(ingredients)
for cuisine in training_data['cuisine']:
    if cuisine not in cuisines:
        cuisines.append(cuisine)
len(cuisines)
X_train=training_data.copy()
X_test=test_data.copy()
for element in ingredients:
    X_train[element]=0
    X_test[element]=0
index =0

for row in X_train['ingredients']:
    for element in row:
        X_train.at[index,element]=1
    index = index + 1 
index =0

for row in X_test['ingredients']:
    for element in row:
        X_test.at[index,element]=1
    index = index + 1 
X_test= X_test.dropna(axis='columns')
X_test= X_test.drop(['id','ingredients'],axis=1)
X_train=X_train.drop(['id','ingredients','cuisine'],axis=1)
Y_train=training_data.copy()
Y_train=Y_train.drop(['id','ingredients'],axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y_train=le.fit_transform(Y_train)

from keras.utils.np_utils import to_categorical
Y_train= to_categorical(Y_train)
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras import optimizers

model_DL=models.Sequential()
model_DL.add(layers.Dense(110,activation='relu',input_shape=(X_train.shape[1],)))
model_DL.add(layers.Dense(80,activation='relu'))
model_DL.add(layers.Dense(20,activation='softmax'))

rmsprop=optimizers.RMSprop(lr=0.001)

model_DL.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])

history=model_DL.fit(X_train,Y_train,epochs=8,batch_size=256)

loss=history.history['loss']

plt.plot(loss,label='Training loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

Y_pred_DL=model_DL.predict(X_test)
prediction_deep=[]
from numpy import argmax

for x in Y_pred_DL:
    prediction_deep.append(np.argmax(x))
    
prediction_deep=le.inverse_transform(prediction_deep)
submission_DL=test_data.copy()
submission_DL=submission_DL.drop(['ingredients'],axis=1)

submission_DL['cuisine']=prediction_deep

submission_DL.to_csv('sample_submission.csv',index=False)