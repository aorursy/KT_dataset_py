# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb



from keras.models import Sequential

from keras.layers import Dense,LSTM,Dropout,LeakyReLU, BatchNormalization

from keras.optimizers import RMSprop, Adadelta, Adam

 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/Stock_Price_Train.csv")

train=df_train.loc[:,["Open"]].values
df_train.head()
df_train.info()
df_train.columns
df_train['Close'] = [x.replace(',', '') for x in df_train['Close']]

df_train['Volume'] = [x.replace(',', '') for x in df_train['Volume']]

df_train['Close']=df_train['Close'].astype(float)

df_train['Volume']=df_train['Volume'].astype(float)
df_train.info()
df_train['Date']=pd.to_datetime(df_train['Date'])

df_train.set_index('Date',inplace=True)

df_train.info()
df_train.isnull().sum()
plt.figure(figsize=(15,12))

sb.lineplot(data=df_train[['Open', 'High', 'Low','Close']],linewidth=2)

plt.grid(True)

plt.show()
plt.style.use('dark_background')

plt.figure(figsize=(15,10))

plt.plot(df_train['Volume'],color='lime',linewidth=2)

plt.grid(True)

plt.xlabel('Volume')

plt.ylabel('Date')

plt.show()
#Normalization

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

train_scaled=scaler.fit_transform(train)

x_train=[]

y_train=[]
for i in range(60,1258):

    x_train.append(train_scaled[i-60:i,0])

    y_train.append(train_scaled[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
opt=RMSprop()    

#opt=Adadelta()    

#opt=Adam()    



time_model=Sequential()

time_model.add(LSTM(units=50,return_sequences=True,activation='tanh',input_shape=(x_train.shape[1],1)))

time_model.add(Dropout(0.2))



time_model.add(LSTM(units=50,activation='tanh',return_sequences=True))

time_model.add(Dropout(0.18))



time_model.add(LSTM(units=50,activation='tanh'))

time_model.add(Dropout(0.15))
time_model.add(Dense(units=1))

time_model.compile(optimizer=opt,loss="mean_squared_error")

time_model.fit(x_train,y_train,epochs=50,batch_size=30)
ds_test=pd.read_csv("../input/Stock_Price_Test.csv")

real_values=ds_test.loc[:,["Open"]].values

ds_total=pd.concat((df_train["Open"],ds_test["Open"]),axis=0)

inputs=ds_total[len(ds_total)-len(ds_test)-60:].values.reshape(-1,1) 

inputs=scaler.transform(inputs)
x_test=[]

for i in range(60,80):

    x_test.append(inputs[i-60:i,0])

x_test=np.array(x_test)

x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predicted_Stock_price=time_model.predict(x_test)

predicted_Stock_price=scaler.inverse_transform(predicted_Stock_price)

plt.figure(figsize=(15,10))

plt.plot(real_values,color="lightgreen",label="Real Stock Values",linewidth=3)

plt.plot(predicted_Stock_price,color="orchid",label="Stock Exchange Predict",linewidth=3)

plt.xlabel("Time")

plt.ylabel("Google Real Values")

plt.grid(True)

plt.legend(prop={'size': 14})

plt.show()