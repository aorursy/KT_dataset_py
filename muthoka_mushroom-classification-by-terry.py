# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/mushrooms.csv")
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in data.columns:

    data[col] = labelencoder.fit_transform(data[col])

 

data.head()
print(data.shape)

data =np.array(data)

#print(data.head)

_length = 8000

train, test = data[0:_length,], data[_length:,]

Xtrain, ytrain = train[:,1:],train[:,0]

Xtest, ytest = test[:,1:], test[:,0]

print(Xtrain.shape)

print(ytrain .shape)

print(Xtest.shape)

print(ytest.shape)

#Import Library

from sklearn import svm

#model = svm.SVC(kernel='linear', c=1, gamma=1) 

model = svm.SVC(kernel='linear', gamma=1) 

# there is various option associated with it, like changing kernel, gamma and C value.

model.fit(Xtrain, ytrain)

scr = model.score(Xtrain, ytrain)

print(scr)

#Predict Output

predicted= model.predict(Xtest)

print(predicted)



dif = predicted - ytest

print(dif)



from keras.models import *

from keras.layers import *



batch_size = 1

mlp_neurons = 5

neurons = 5

bi_neurons = 5

repeats = 5

nb_epochs = 5

## mlp

def mlp_model(train, batch_size, nb_epoch, neurons):

    X,y = train[:,1:], train[:,0]

    #X = X.reshape(X.shape[0],1,X.reshape[1])

    model = Sequential()

    model.add(Dense(neurons,input_dim = X.shape[1],init = 'normal', activation = 'relu'))

    model.add(Dropout(0.2))

    model.add(Dense(neurons, init = 'normal', activation = 'relu'))

    model.add(Dropout(0.2))

    model.add(Dense(neurons, init = 'normal', activation = 'relu'))

    model.add(Dense(1, activation = 'linear'))

    model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])

    for i in range(nb_epoch):

        model.fit(X,y, epochs = 1, batch_size = batch_size, verbose = 2, shuffle = False)

        model.reset_states()

    return model
def forecast_mlp(model, batch_size, row):

    X = row

    X = X.reshape(1,len(X))

    yhat = model.predict(X, batch_size = batch_size)

    #return yhat[0,0]

    return yhat
mlp_RNN = mlp_model(train, batch_size, nb_epochs, mlp_neurons)
def simulated_mlp(model, train, batch_size, nb_epochs, neurons):

    n1 = len(Xtest)

    n2 = repeats

    predictions1 = np.zeros((n1,n2), dtype = float)

    for r in range(repeats):

        predictions2 = list()

        for i in range(len(Xtest)):

            if(i == 0):

               y = forecast_mlp(model, batch_size, Xtest[i,:])

               Xtest[i+1,:-1] = Xtest[i,1:]

               Xtest[i+1,-1] = y

               predictions2.append(y)

            else:

               y = forecast_mlp(model, batch_size, Xtest[i-1,:])

               Xtest[i,:-1] = Xtest[i-1,1:]

               Xtest[i,-1] = y

               predictions2.append(y)

        predictions1[:,r] = predictions2

    return np.mean(predictions1, axis = 1)
print(ytest)

print("===================== mlp ==================================")

print(simulated_mlp(mlp_RNN, train, batch_size, nb_epochs, mlp_neurons))