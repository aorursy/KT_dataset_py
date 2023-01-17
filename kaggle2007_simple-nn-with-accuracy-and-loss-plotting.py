# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pan

from sklearn.preprocessing import StandardScaler

import sklearn.preprocessing as pre

import keras as ks

import matplotlib.pyplot as plt

from keras.layers import Dense

from keras.callbacks import EarlyStopping

from keras.regularizers import l2

from keras.layers import Dropout

from keras import optimizers
#Given: Train and Test Data set - separately



#Reading Train Data set
def readfile(filename):

    filename = "../input/" + filename

    return(pan.read_csv(filename))
def checkfornull(dataset):

    return(dataset.columns[dataset.isnull().any()])
data_set_train = readfile('train.csv')
TrainData_X = data_set_train.drop(['ID_code', 'target'], axis=1)

TargetTrainData_Y = data_set_train['target']
data_set_testing = readfile("test.csv")

TestData_X = data_set_testing.drop(['ID_code'], axis=1) # excluding ID_code, 200 columns
# Check for null or NaN values



null_columns=checkfornull(data_set_train)

if(null_columns.size == 0):

    print("There are no columns with NULL values in the Training Data set.")
null_columns = checkfornull(data_set_testing)

if(null_columns.size == 0):

    print("There are no columns with NULL values in the Testing Data set.")
# Standardized and scaled data

scaler = StandardScaler()

scaled_train_data_X = scaler.fit_transform(TrainData_X)

scaled_test_data_X = scaler.transform(TestData_X)
#OneHotCoding



oneHot = pre.OneHotEncoder(sparse=False)



b = np.array(TargetTrainData_Y)

TargetHot = b.reshape((len(b)), -1)

TargetHot_encoded = oneHot.fit_transform(TargetHot)
#Defining the Neural Network Model



model = ks.Sequential()

model.add(Dense(90, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.001)))

model.add(Dropout(0.1))

model.add(Dense(50, activation='sigmoid', kernel_regularizer=l2(0.001)))

model.add(Dense(2, activation='relu'))
print("---- compiling the model ----\n")

opti = optimizers.Adamax(beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

model.compile(loss='mean_squared_error', optimizer=opti,  metrics=['accuracy'])
# simple early stopping

print("---- Early stopping, optimized for val_loss ----\n")

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
print("---- running the model.fit ----\n")

history=model.fit(scaled_train_data_X, TargetHot_encoded, validation_split=0.2, \

                  callbacks=[es], epochs=100, batch_size=2200, shuffle=False, verbose=0)
# evaluate the model

_, train_acc = model.evaluate(scaled_train_data_X, TargetHot_encoded, verbose=0)

print('Train Accuracy: %.3f' % (train_acc))
# plot the accuracy and loss

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.title('Plot History: Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='test loss')

plt.title('Plot History: Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
predict = model.predict(scaled_test_data_X)



result = pan.DataFrame({"ID_code": data_set_testing['ID_code'], "target": predict[:,0]})

print(result.head())



result.to_csv("submission.Hemal.Mar102019.1.csv", index=False)