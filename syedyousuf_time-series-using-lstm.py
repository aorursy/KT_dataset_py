# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv',parse_dates=['Time Serie'],index_col=1)
df.drop('Unnamed: 0',axis=1,inplace=True)
df.info()
df.dtypes
df.isna().sum()
df=df.replace('ND',0)
df
for dtype in df.dtypes:

    df=df.astype('float')
df.dtypes
fig1,(ax1,ax2)=plt.subplots(2,1,sharex=True)

df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'].plot(ax=ax1)

df['EURO AREA - EURO/US$'].plot(ax=ax2)





ax1.set_title('AUSTRALIA - AUSTRALIAN DOLLAR/US$',loc='right')

ax2.set_title('EURO AREA - EURO/US$',loc='right')







fig2,(ax3,ax4)=plt.subplots(2,sharex=True,)

df['NEW ZEALAND - NEW ZELAND DOLLAR/US$'].plot(ax=ax3,)

df['UNITED KINGDOM - UNITED KINGDOM POUND/US$'].plot(ax=ax4)





ax3.set_title('NEW ZEALAND - NEW ZELAND DOLLAR/US$',loc='right')

ax4.set_title('UNITED KINGDOM - UNITED KINGDOM POUND/US$',loc='right')



plt.show()
len(df)
df.tail()
ml_df=df.loc['2018-01-01':]
ml_df=ml_df.round(2)
ml_df
test_ind=5
train=ml_df.iloc[:-test_ind]

test=ml_df.iloc[-test_ind:]
train
test
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(train)
scaled_train=scaler.transform(train)

scaled_test=scaler.transform(test)
scaled_train.shape
from keras.preprocessing.sequence import TimeseriesGenerator
length=2

generator=TimeseriesGenerator(scaled_train,scaled_train,batch_size=1,length=length)
val_generator=TimeseriesGenerator(scaled_test,scaled_test,batch_size=1,length=length)
from keras.models import Sequential

from keras.layers import Dense,LSTM
n_features=scaled_train.shape[1]

model=Sequential()

model.add(LSTM(100,activation='relu',input_shape=(length,n_features)))

model.add(Dense(n_features))

model.compile(optimizer='adam',loss='mse')
model.summary()
from keras.callbacks import EarlyStopping
early=EarlyStopping(monitor='val_loss',patience=3)
model.fit_generator(generator,epochs=10,callbacks=[early],validation_data=val_generator)
losses=pd.DataFrame(model.history.history)
losses.plot()
prediction=model.predict_generator(val_generator)
true_prediction=scaler.inverse_transform(prediction)
test_df=test[2:]
test_df
prediction_df=pd.DataFrame(true_prediction,columns=test.columns,index=test_df.index)
prediction_df
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,6),sharex=True)

ax1.plot(test_df)

ax2.plot(prediction_df)



ax1.set_title('True Value')

ax2.set_title('Predicted Value')
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test_df['NEW ZEALAND - NEW ZELAND DOLLAR/US$'],prediction_df['NEW ZEALAND - NEW ZELAND DOLLAR/US$']))