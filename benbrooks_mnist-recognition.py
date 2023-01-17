# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

#organizing data
train = pd.read_csv("../input/train.csv")
test= pd.read_csv("../input/test.csv")
test = test.astype('float32')
y_train = train['label'].astype('float32')
x_train = train.drop('label',axis=1).astype('int32')

#pre-processing
y_train= to_categorical(y_train)

#model construction
model= Sequential()
model.add(Dense(128, input_dim=784, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, epochs=10, batch_size=1500)

results = model.predict_classes(test)
print(results.shape)

my_submission=pd.DataFrame({"ImageId": list(range(1,len(results)+1)), "Label": results})
my_submission.to_csv('submission.csv', index=False)

# Any results you write to the current directory are saved as output.
