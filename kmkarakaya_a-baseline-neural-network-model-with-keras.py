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
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# load train dataset
dataframe = pandas.read_csv("../input/train.csv")
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
y = dataset[:,6:8]

#print("X: first 4 rows")
#print(X[:4])
#print("Y: first 4 rows")
#print(y[:4])
# load test dataset
testdataframe = pandas.read_csv("../input/test.csv")
testdataset = testdataframe.values
# split into input (X) and output (Y) variables
Xtest = testdataset[:,1:7]

print("Xtest: first 4 rows")
print(Xtest[:4])

def createModel(input,output):
    # define and fit the final model
    model = Sequential()
    model.add(Dense(input.shape[1], input_dim=6, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output.shape[1], activation='linear'))
    model.compile(loss='mae', optimizer='RMSprop', metrics=['mae'])

    model.summary()
    return model


def fitPredict(model, X, y, Xtest):
    #Normalize data
    scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
    scalarX.fit(X)
    scalarY.fit(y)
    X = scalarX.transform(X)
    y = scalarY.transform(y)

    '''
    print("X: first 4 rows")
    print(X[:4])
    print("Y: first 4 rows")
    print(y[:4])
    '''
    
    # new instances where we do not know the answer
    print("Xtest: first 4 rows")
    print(Xtest[:4])
    Xtest = scalarX.transform(Xtest)
    

    
    
    
    
    model.fit(X, y, epochs=1, verbose=0, validation_split=0.20, batch_size=20)
    
    
    
    # make a prediction
    ytest = model.predict(Xtest)
    ytest= scalarY.inverse_transform(ytest)
    xtest = scalarX.inverse_transform(Xtest)
    # show the inputs and predicted outputs
    for i in range(4):
        print("X=%s, Predicted=%s" % (xtest[i], ytest[i]))
        
    return ytest

model = createModel(X,y)
prediction = fitPredict(model,X,y,Xtest)


a= prediction[:,0]
b = prediction[:,1]

my_submission = pd.DataFrame({ 'ID': testdataframe.ID,'A': a, 'B': b})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
my_submission.head()
