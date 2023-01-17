import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential

import matplotlib.patches as mpatches

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error





import os





# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/testset.csv')
data.columns
data.head()
data['datetime_utc'] = pd.to_datetime(data['datetime_utc'])

data.set_index('datetime_utc', inplace= True)

data =data.resample('D').mean()
data = data[[' _tempm' ]]
data.info()
data[' _tempm'].fillna(data[' _tempm'].mean(), inplace=True) # we will fill the null row
data.info()
data.head()
plt.figure(figsize=(20,8))

plt.plot(data)

plt.title('Time Series')

plt.xlabel('Date')

plt.ylabel('temperature')

plt.show()
data=data.values

data = data.astype('float32')
scaler= MinMaxScaler(feature_range=(-1,1))

sc = scaler.fit_transform(data)
timestep = 30



X= []

Y=[]





for i in range(len(sc)- (timestep)):

    X.append(sc[i:i+timestep])

    Y.append(sc[i+timestep])





X=np.asanyarray(X)

Y=np.asanyarray(Y)





k = 7300

Xtrain = X[:k,:,:]

Xtest = X[k:,:,:]    

Ytrain = Y[:k]    

Ytest= Y[k:]   
print(Xtrain.shape)

print(Xtest.shape)
from keras.layers import Dense,RepeatVector

from keras.layers import Flatten

from keras.layers import TimeDistributed

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D
model = Sequential()

model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(30,1)))

model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(RepeatVector(30))

model.add(LSTM(128, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(Xtrain,Ytrain,epochs=300, verbose=0 )
preds_cnn1 = model.predict(Xtest)

preds_cnn1 = scaler.inverse_transform(preds_cnn1)





Ytest=np.asanyarray(Ytest)  

Ytest=Ytest.reshape(-1,1) 

Ytest = scaler.inverse_transform(Ytest)





Ytrain=np.asanyarray(Ytrain)  

Ytrain=Ytrain.reshape(-1,1) 

Ytrain = scaler.inverse_transform(Ytrain)



mean_squared_error(Ytest,preds_cnn1)
plt.figure(figsize=(20,9))

plt.plot(Ytest , 'blue', linewidth=5)

plt.plot(preds_cnn1,'r' , linewidth=4)

plt.legend(('Test','Predicted'))

plt.show()
def insert_end(Xin,new_input):

    for i in range(timestep-1):

        Xin[:,i,:] = Xin[:,i+1,:]

    Xin[:,timestep-1,:] = new_input

    return Xin
first =0   # this section for unknown future 

future=200

forcast_cnn = []

Xin = Xtest[first:first+1,:,:]

for i in range(future):

    out = model.predict(Xin, batch_size=1)    

    forcast_cnn.append(out[0,0]) 

    Xin = insert_end(Xin,out[0,0]) 
forcasted_output_cnn=np.asanyarray(forcast_cnn)   

forcasted_output_cnn=forcasted_output_cnn.reshape(-1,1) 

forcasted_output_cnn = scaler.inverse_transform(forcasted_output_cnn) 
plt.figure(figsize=(16,9))

plt.plot(Ytest , 'black', linewidth=4)

plt.plot(forcasted_output_cnn,'r' , linewidth=4)

plt.legend(('test','Forcasted'))

plt.show()