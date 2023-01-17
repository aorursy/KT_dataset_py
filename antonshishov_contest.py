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
import tensorflow as tf

import pandas as pd

import numpy as np

from tensorflow import keras

from sklearn.metrics import f1_score

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Activation

import keras.backend as K

import csv

import seaborn as sns

import scikitplot as skplt





def f1(y_true, y_pred):

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)



def attackType(arr):

    if round(arr[0]) == 1.0:

        return 'BENIGN'

    if round(arr[1]) == 1.0:

        return 'WebAttackBF'

    if round(arr[2]) == 1.0:

        return 'WebAttackSql'

    if round(arr[3]) == 1.0:

        return 'WebAttackXSS'

    return 'BENIGN'



dataSet = pd.read_csv(filepath_or_buffer = '/kaggle/input/ids-minor-web-2019/TWHM-WebAttacksTrain.csv', low_memory = False)

dataSetDropped = dataSet.drop(columns = ['Source IP', 'Destination IP', 'Flow ID','Flow Bytes/s', 'Flow Packets/s', 'Timestamp', 'Usage'])

dataSetModified = pd.get_dummies(data = dataSetDropped,columns = ['Label'],dtype = int)

sns.heatmap(dataSet.corr())

dataFrame = dataSetModified.to_numpy(dtype = float) 

inputMatrix = dataFrame[:, 0:78]

outputMatrix = dataFrame[:, 79:83]



model = Sequential([

    Dense(79, activation='sigmoid'),

    Dense(79, activation='sigmoid'),

    Dense(4, input_shape=(4,), activation='softmax'),

])



model.compile(loss='mean_squared_error', optimizer='SGD', metrics=[f1])

model.fit(inputMatrix, outputMatrix, epochs=20, batch_size=30)

_, accuracy = model.evaluate(inputMatrix, outputMatrix)

predictions = model.predict(inputMatrix)

print(accuracy)



resultArray = ['IDNum,Label']

for i in range(predictions.__len__()):

    attackVil = attackType(predictions[i])

    idNum = str(dataSet['IDNum'][i])

    stringToAdd = '{0},{1}'.format(idNum, attackVil)

    resultArray.append(stringToAdd)



dataSetPredict = pd.read_csv(filepath_or_buffer = '/kaggle/input/ids-minor-web-2019/TWHM-WebAttacksTestPrivat.csv', low_memory = False)

dataSetDroppedPredict = dataSetPredict.drop(columns = ['Source IP', 'Destination IP', 'Flow ID','Flow Bytes/s', 'Flow Packets/s', 'Timestamp', 'Usage'])

dataSetModifiedPredict = dataSetDroppedPredict



dataFramePredict = dataSetModifiedPredict.to_numpy(dtype = float)[:,0:78]



predictionsNew = model.predict(dataFramePredict)

for i in range(len(predictions)):

  for j in range(len(predictions[i])):

    predictions[i][j] = round(predictions[i][j])



skplt.metrics.plot_confusion_matrix(

    outputMatrix.argmax(axis=1), 

    predictions.argmax(axis=1))



for i in range(predictionsNew.__len__()):

    attackVil = attackType(predictionsNew[i])

    idNum = str(dataSetPredict['IDNum'][i])

    stringToAdd = '{0},{1}'.format(idNum, attackVil)

    resultArray.append(stringToAdd)

    

pd.DataFrame(resultArray).to_csv('/kaggle/working/result1.csv',index = False, quoting=csv.QUOTE_NONNUMERIC)

print('saved to csv')