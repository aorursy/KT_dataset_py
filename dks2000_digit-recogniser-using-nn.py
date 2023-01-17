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
import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

import math

import seaborn as sns

from IPython import display

from matplotlib import cm

from matplotlib import gridspec

from matplotlib import pyplot as plt

import time
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print (train.describe())					#Gives statitics of the data



X_train = train.loc[:, train.columns != 'label']

y_labels_train = train["label"]

# print (X_train)

# print (y_labels_train)
print (test.describe())					#Gives statitics of the data



X_test = test.loc[:, test.columns != 'label']

# print (X_test)
X_train = X_train.values

X_test = X_test.values



y_labels_train = y_labels_train.values

lb = preprocessing.LabelBinarizer()

lb.fit(y_labels_train)

output_classes = lb.classes_

print ("No.of Output Classes = ",output_classes)

y_train = lb.transform(y_labels_train)
Data_y = {}



for i in output_classes:

    Data_y[i] = 0

for j in y_labels_train:

    Data_y[j] += 1

    

print (Data_y)

plt_ = sns.barplot(list(Data_y.keys()), list(Data_y.values()))

plt_.set_xticklabels(plt_.get_xticklabels(), rotation=90)

plt.show()
print ("Shape of Training Set is",X_train.shape)

print ("Shape of Test Set is",X_test.shape)



print ("Shape of Training Set is",y_train.shape)
keras.backend.clear_session()

model = Sequential()

model.add(Dense(1024, input_dim=784, activation='sigmoid'))

model.add(Dense(1024, activation='sigmoid'))

model.add(Dense(128, activation='sigmoid'))

model.add(Dense(128, activation='sigmoid'))

model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=64, validation_split = 0.1, batch_size = 2048)
result = model.predict(X_test)

predictions = np.apply_along_axis(lambda row: np.argmax(row),1,result)

predictions.reshape([-1,1])
print (predictions)
output_test_data = pd.DataFrame() 

output_test_data['Label'] = predictions

rows = predictions.shape[0]

print (rows)

output_test_data['ImageId'] = list(np.arange(1,rows+1))

submission = output_test_data[['ImageId','Label']]

submission.to_csv("submission.csv", index=False)

submission.tail()