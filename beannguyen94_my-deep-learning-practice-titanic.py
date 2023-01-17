# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
def create_dataset(dataset, istrain=True):
    # drop NaN
    dataset = dataset.fillna(0)
    # convert sex: male = 0, female = 1
    dataset['Sex'] = [0 if r == 'male' else 1 for r in dataset['Sex']]
    
    # features and labels list
    dataX, dataY = [], []
    for row in dataset.values:
        if istrain:
            dataX.append(row[1:])
            dataY.append(row[0])
        else:
            dataX.append(row)
        
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    dataX = np.array(x_scaler.fit_transform(dataX))
    return dataX, dataY
    

trainset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')
testY = pd.read_csv('../input/gender_submission.csv')

# drop unnessary cols
trainset.drop(trainset.columns[[0, 3, 8, 10, 11]], axis=1, inplace=True)
testset.drop(testset.columns[[0, 2, 7, 9, 10]], axis=1, inplace=True)
testY.drop(testY.columns[[0]], axis=1, inplace=True)

# data preprocessing
trainX, trainY = create_dataset(trainset, True)
testX, _ = create_dataset(testset, False)
nb_features = 6

def create_model():
    model = Sequential()
    model.add(Dense(32, input_dim=nb_features, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
batch_size = 128
nb_epochs = 200

# print(trainY)
model = create_model()
model.fit(trainX, trainY,
          epochs=nb_epochs,
          batch_size=batch_size,
          verbose=0)
score = model.evaluate(testX, testY, batch_size=batch_size)
print(score)
import csv

headers = ['PassengerId', 'Survived']
testset = pd.read_csv('../input/test.csv')
y_test = model.predict_classes(testX)

with open('predicted.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    writer.writeheader()
    i = 0
    for row in testset['PassengerId']:
        writer.writerow({
            'PassengerId': row,
            'Survived': y_test[i][0]
        })
        i += 1
