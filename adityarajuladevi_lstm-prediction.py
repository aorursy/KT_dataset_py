# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import LSTM,Dense

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#The point of this is to fit a RNN to the dataset to generate the next close price of the dataset

data = pd.read_csv('../input/all_stocks_5yr.csv')

data.head()

#We will be using the closing data of MMM(want to do something other than AAPL :))

cl = data[data['Name']=='GOOG'].Close

scl = MinMaxScaler()

#Scale the data

cl = cl.reshape(cl.shape[0],1)

cl = scl.fit_transform(cl)

cl
#Create a function to process the data into 7 day look back slices

def processData(data,lb):

    X,Y = [],[]

    for i in range(len(data)-lb-1):

        X.append(data[i:(i+lb),0])

        Y.append(data[(i+lb),0])

    return np.array(X),np.array(Y)

X,y = processData(cl,7)

X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]

y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]

print(X_train.shape[0])

print(X_test.shape[0])

print(y_train.shape[0])

print(y_test.shape[0])
#Build the model

model = Sequential()

model.add(LSTM(256,input_shape=(7,1)))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

#Reshape data for (Sample,Timestep,Features) 

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))

X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

#Fit model with history to check for overfitting

history = model.fit(X_train,y_train,epochs=300,validation_data=(X_test,y_test),shuffle=False)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend(['train','validation'])
#We see this is pretty jumpy but we will keep it at 300 epochs. With more data, it should smooth out the loss

#Lets look at the fit

Xt = model.predict(X_test)

plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))

plt.plot(scl.inverse_transform(Xt))



#This looks good for finding a trend

act = []

pred = []

for i in range(4):

    Xt = model.predict(X_test[i].reshape(1,7,1))

    print('predicted:{0}, actual:{1}'.format(scl.inverse_transform(Xt),scl.inverse_transform(y_test[i].reshape(-1,1))))

    pred.append(scl.inverse_transform(Xt))

    act.append(scl.inverse_transform(y_test[i].reshape(-1,1)))



#Lets try with a stacked LSTM just for fun

#Build the model

model = Sequential()

model.add(LSTM(256,return_sequences=True,input_shape=(7,1)))

model.add(LSTM(256))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

#Reshape data for (Sample,Timestep,Features) 

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))

X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

#Fit model with history to check for overfitting

history = model.fit(X_train,y_train,epochs=300,validation_data=(X_test,y_test),shuffle=False)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend(['train','validation'])
#Much cleaner loss this time, still could use some more data. Ideally should run multiple to get

#the average loss/val loss plots. Next time would probably stop at 100 epochs

Xt = model.predict(X_test)

plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))

plt.plot(scl.inverse_transform(Xt))
