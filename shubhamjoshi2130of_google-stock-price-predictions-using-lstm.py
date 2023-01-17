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
stock_price=pd.read_csv("../input/trainset.csv")


stock_price_train=stock_price.iloc[:,1:2].values

stock_price.head()
stock_price_train.shape
from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler(feature_range=(0,1))



stock_price_scaled=sc.fit_transform(stock_price_train)

pd.DataFrame(stock_price_scaled).head()
X_train=[]

y_train=[]



for i in range(60,1258):

    X_train.append(stock_price_scaled[i-60:i,0])

    y_train.append(stock_price_scaled[i,0])

    

X_train,y_train=np.array(X_train),np.array(y_train)



pd.DataFrame(X_train).head()

X_train.shape
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_train.shape
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout



# Initializing RMM

regressor=Sequential()



# Adding the first LSTM layer and some Dropout regularization

regressor.add(LSTM(units = 50,return_sequences=True,input_shape=(X_train.shape[1],1)))

regressor.add(Dropout(0.2))



# 2nd LSTM layer

regressor.add(LSTM(units = 50,return_sequences=True))

regressor.add(Dropout(0.2))



# 3 LSTM Layer

regressor.add(LSTM(units = 50,return_sequences=True))

regressor.add(Dropout(0.2))



# 4 LSTM Layer

regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))





# Adding the output layer

regressor.add(Dense(units=1))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fitting the RNN to training set

regressor.fit(X_train,y_train,epochs=100, batch_size=32)
# reading the test data

stock_price_test=pd.read_csv("../input/testset.csv")

real_stock_price=stock_price.iloc[:,1:2].values

import matplotlib.pyplot as plt

import seaborn as sns

b=sns.lineplot(x="Date",y="Open",data=pd.DataFrame(stock_price))

b.set_xticklabels([],rotation=90)





# Lets predict the stock price of january 2017



dataset_total=pd.concat((stock_price['Open'],stock_price_test['Open']),axis=0)

inputs=dataset_total[len(stock_price)-len(stock_price_test)-125:].values

inputs=inputs.reshape(-1,1)

inputs=sc.transform(inputs)



len(stock_price_test),len(inputs)
X_test=[]

for i in range(60,125):

    X_test.append(inputs[i-60:i,0])

X_test=np.array(X_test)



X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))



predicted_stock_price = regressor.predict(X_test)



predicted_stock_price=sc.inverse_transform(predicted_stock_price)



plt.plot(real_stock_price,color='red',label='Real Google Stock Price')

plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()