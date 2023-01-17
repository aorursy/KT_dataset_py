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
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load train dataset
trainDataFrame = pandas.read_csv("../input/trainSimple.csv")
trainDataFrame.describe()



# split into input (X) and output (Y) variables
X = trainDataFrame.iloc[:,0:6]
y = trainDataFrame.iloc[:,6:8]

print(X.head())
print(y.head())
X=X.values
y=y.values
# create training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state=1)
print (X_train.shape, y_train.shape)
print (X_validation.shape, y_validation.shape)
# load test dataset
testDataFrame = pandas.read_csv("../input/testSimple.csv")
testDataFrame.describe()

# remove ID column for converting to input (X)
Xtest = testDataFrame.iloc[:,1:8]


print(Xtest.head())

Xtest= Xtest.values
def showHistory(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model accuracy')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
def transformData ( X_train, X_validation, y_train, y_validation, Xtest):
    #Normalize data
    global scalarX, scalarY 
    scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
    #scalarX, scalarY = RobustScaler(), RobustScaler()
    #scalarX, scalarY = StandardScaler(), StandardScaler()
    
    
    X_train, X_validation, y_train, y_validation
    
    scalarX.fit(X_train)
    scalarY.fit(y_train)
    X_train = scalarX.transform(X_train)
    y_train = scalarY.transform(y_train)
    
    X_validation = scalarX.transform(X_validation)
    y_validation = scalarY.transform(y_validation)
        
    Xtest = scalarX.transform(Xtest)
    
    return X_train, X_validation, y_train, y_validation,Xtest
def inverseTransformData (ytest):
    ytest= scalarY.inverse_transform(ytest)
    return ytest
    
def createModel(input,output):
    # define and fit the final model
    model = Sequential()
    model.add(Dense(input.shape[1], input_dim=6, activation='relu'))
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(512, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output.shape[1], activation='linear'))
    model.compile(loss='MSE', optimizer='adam', metrics=['mae'])

    model.summary()
    return model


def fitPredict(model, X_train, X_validation, y_train, y_validation,Xtest):
    
    #history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_split=0.10, batch_size=25)
    history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_validation,y_validation), batch_size=25)
    
     
    
    # make a prediction
    ytest = model.predict(Xtest)
    
    
    
    
        
    return ytest , history
a=[[0.0002,0.0000004],
   [4.9,-2000],
   [5,50000000]]


scalarA= RobustScaler()
#RobustScaler() StandardScaler() MinMaxScaler
    
scalarA.fit(a)
aNew = scalarA.transform(a)
print(a)
print( aNew)

v= pd.DataFrame(aNew)
print(v.describe())
v.head()

aInv = scalarA.inverse_transform(aNew)
print( aInv)




X_train, X_validation, y_train, y_validation,Xtest = transformData(X_train, X_validation, y_train, y_validation,Xtest)


model = createModel(X_train,y_train)
predictions, trainingHistory = fitPredict(model,X_train, X_validation, y_train, y_validation,Xtest)
predictions = inverseTransformData(predictions)

showHistory(trainingHistory)

predictions
a= predictions[:,0]
b = predictions[:,1]

my_submission = pd.DataFrame({ 'ID': testDataFrame.ID,'A': a, 'B': b})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
my_submission.head()
