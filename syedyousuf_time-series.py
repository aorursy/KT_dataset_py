# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv',parse_dates=['Time Serie'],index_col='Time Serie')
df.head()
df.drop('Unnamed: 0',axis=1,inplace=True)
df.info()
df.head()
df.dtypes
df=df.replace('ND','-1')
for c in df.columns:

    if df[c].dtype=='object':

        df[c]=df[c].astype('float')
df.dtypes
import matplotlib.pyplot as plt

import seaborn as sns
fig,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,sharex=True,figsize=(10,10))

df.plot(y='AUSTRALIA - AUSTRALIAN DOLLAR/US$',ax=ax1,legend=False)

ax1.set_title('AUSTRALIA - AUSTRALIAN DOLLAR/US',y=1,loc='right')



df.plot(y='EURO AREA - EURO/US$',ax=ax2,legend=False)

ax2.set_title('EURO AREA - EURO/US$',y=1,loc='right')



df.plot(y='NEW ZEALAND - NEW ZELAND DOLLAR/US$',ax=ax3,legend=False)

ax3.set_title('NEW ZEALAND - NEW ZELAND DOLLAR/US$',y=1,loc='right')



df.plot(y='UNITED KINGDOM - UNITED KINGDOM POUND/US$',ax=ax4,legend=False)

ax4.set_title('UNITED KINGDOM - UNITED KINGDOM POUND/US$',y=1,loc='right')



plt.show()
test_perc=30
test_ind=test_perc*1
train=df.iloc[:-test_ind]

test=df.iloc[-test_ind:]
train
test
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(train)
scaled_train=scaler.transform(train)

scaled_test=scaler.transform(test)
scaled_train
scaled_test
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
length=3

generator=TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=1)
n_features=scaled_train.shape[1]

model=Sequential()

model.add(LSTM(100,activation='relu',input_shape=(length,n_features),return_sequences=True))

model.add(LSTM(100,activation='relu'))

model.add(Dense(n_features))
model.compile(optimizer='adam',loss='mse')
val_generator=TimeseriesGenerator(scaled_test,scaled_test,length=length,batch_size=1)
from tensorflow.keras.callbacks import EarlyStopping
early=EarlyStopping(monitor='val_loss',patience=5)
model.fit_generator(generator,epochs=20,validation_data=val_generator,callbacks=[early])
losses=pd.DataFrame(model.history.history)
losses.plot()
prediction=model.predict_generator(val_generator)
prediction
true_predictions=scaler.inverse_transform(prediction)
prediction_df=pd.DataFrame(true_predictions,index=test.index[3:],columns=test.columns)
prediction_df
test_df=test.iloc[3:,:]
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test_df['TAIWAN - NEW TAIWAN DOLLAR/US$'],prediction_df['TAIWAN - NEW TAIWAN DOLLAR/US$']))