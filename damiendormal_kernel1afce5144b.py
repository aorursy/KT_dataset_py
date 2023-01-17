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
import tensorflow as tf

from tensorflow.keras import layers

from keras.models import Model

from keras.layers import Input, Dense



originalTrainDataFrame = pd.read_csv('../input/train.csv')

trainDataFrame = originalTrainDataFrame
originalTrainDataFrame
labelsDataFrame = trainDataFrame[['Survived']].copy()

trainDataFrame = trainDataFrame.drop(labels=['Survived'], axis=1)

ageMean = trainDataFrame.loc[:,'Age'].mean()

fareMean = trainDataFrame.loc[:,'Fare'].mean()
def adapt_dataFrame(dataFrame, ageMean, fareMean):



    # replace null ages by age mean

    dataFrame = dataFrame.fillna({"Age": ageMean})



    #replace Cabin null by 0

    dataFrame = dataFrame.fillna({"Cabin": 0})



    #create cabinBool Column & insert into

    dataFrame['CabinBool'] = np.where(dataFrame['Cabin'] == 0, 0, 1)

    

    # replace null fare by fare mean

    fareMean = dataFrame.loc[:,'Fare'].mean()

    dataFrame = dataFrame.fillna({"Fare": fareMean})



    # replace null embarked by max column value

    dataFrame = dataFrame.fillna({"Embarked": 'S'})

    #order of embarkation

    #After leaving Southampton on 10 April 1912, Titanic called at Cherbourg in France and Queenstown (now Cobh) in Ireland

    #create PortCategory Column & insert into

    dataFrame['EmbarkedCat'] = np.where(dataFrame['Embarked'] == 'S', 1, np.where(dataFrame['Embarked'] == 'C', 2, 3))



    #create sexBool Column & insert into

    dataFrame['SexBool'] = np.where(dataFrame['Sex'] == 'male', 0, 1)



    # remove columns

    dataFrame = dataFrame.drop(labels=['PassengerId', 'Name', 'Cabin', 'Sex', 'Ticket', 'Embarked'], axis=1)

    

    return dataFrame



    #to do

    #add title of the person
# display null value in the data frame

# trainDataFrame[trainDataFrame.isnull().any(axis=1)]



# display data frame

# trainDataFrame



# type of data

# trainDataFrame.dtypes
trainDataFrame = adapt_dataFrame(trainDataFrame, ageMean, fareMean)

trainDataArray = trainDataFrame.values

trainLabelsArray = labelsDataFrame.values

inputDims = np.size(trainDataArray,1)
#inputs = Input(shape=(np.size(trainArray,1),)) # input layer

inputs = Input(shape=(inputDims,)) # input layer

x = Dense(32, activation='relu')(inputs) # hidden layer

outputs = Dense(1, activation='sigmoid')(x) #output layer



model = Model(inputs, outputs)
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
model.summary()
model.fit(x=trainDataArray, y=trainLabelsArray, batch_size=20, epochs=200)
#trainDataFrame['EmbarkedCat'].value_counts()
originalTestDataFrame = pd.read_csv('../input/test.csv')

testDataFrame = originalTestDataFrame



testLabelsDataFrame = pd.read_csv('../input/gender_submission.csv')

testLabelsDataFrame = testLabelsDataFrame.drop(labels=['PassengerId'], axis=1)

testLabelsDataFrame.dtypes
testDataFrame = adapt_dataFrame(testDataFrame, ageMean, fareMean)
# display null values

# testDataFrame[testDataFrame.isnull().any(axis=1)]



# metric values returned when the model is evaluated

# model.metrics_names
testDataArray = testDataFrame.values

testLabelsArray = testLabelsDataFrame.values
loss, acc = model.evaluate(x=testDataArray, y=testLabelsArray)

print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
y_pred = model.predict(x=testDataArray)

y_pred = (y_pred > 0.5).astype(int)

yDataFrame = pd.DataFrame(columns=['y'])

yDataFrame['y'] = y_pred[:,0]



data_to_submit = pd.DataFrame({

    'PassengerId':originalTestDataFrame['PassengerId'],

    'Survived':yDataFrame['y']

})
data_to_submit
data_to_submit.to_csv('csv_to_submit.csv', index = False)