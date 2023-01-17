import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





from keras.models import Sequential

from keras.layers import Dense,RepeatVector

from keras.layers import Flatten

from keras.layers import TimeDistributed

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D







from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



import os

print(os.listdir("../input"))

data=pd.read_csv('../input/sales_train.csv')

data.info()
data.head()
data['date'] = pd.to_datetime(data['date'])
data.info()
data.set_index(['date'],inplace=True)

data = data['item_cnt_day'].resample('D').sum()

df=pd.DataFrame(data)
plt.figure(figsize=(16,8))

df['item_cnt_day'].plot()

plt.xlabel('Date')

plt.ylabel('Number of Products Sold')

plt.show()
df_1=df.values

df_1=df_1.astype('float32')



scaler = MinMaxScaler(feature_range=(-1,1))

ts = scaler.fit_transform(df_1)
df.info()
timestep = 30



X= []

Y=[]



raw_data=ts



for i in range(len(raw_data)- (timestep)):

    X.append(raw_data[i:i+timestep])

    Y.append(raw_data[i+timestep])





X=np.asanyarray(X)

Y=np.asanyarray(Y)





k = 850

Xtrain = X[:k,:,:]  

Ytrain = Y[:k]    
model = Sequential()

model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(30, 1)))

model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))

model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(Xtrain, Ytrain, epochs=200, verbose=0)
Xtest = X[k:,:,:]  

Ytest= Y[k:]  
preds = model.predict(Xtest)

preds = scaler.inverse_transform(preds)





Ytest=np.asanyarray(Ytest)  

Ytest=Ytest.reshape(-1,1) 

Ytest = scaler.inverse_transform(Ytest)





Ytrain=np.asanyarray(Ytrain)  

Ytrain=Ytrain.reshape(-1,1) 

Ytrain = scaler.inverse_transform(Ytrain)



mean_squared_error(Ytest,preds)
from matplotlib import pyplot

pyplot.figure(figsize=(20,10))

pyplot.plot(Ytest)

pyplot.plot(preds, 'r')

pyplot.show()