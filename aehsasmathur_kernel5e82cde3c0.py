# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
### Training set data
data=pd.read_csv('/kaggle/input/bovespa-index-ibovespa-stocks-data/ABEV3_series.csv')
data.head()
data.isnull().sum() ## checking for data with null values
plt.figure(num='Date',figsize=(10,4))
plt.plot(data['open'])
plt.plot(data['close'])
plt.plot(data['high'])
plt.plot(data['low'])
plt.xlabel('days')
plt.ylabel('price (USD)')
plt.title('stock price history')
plt.legend(['open','high','close','low'])
plt.show()
plt.figure(num='Date' , figsize=(10,4))
plt.plot(data['volume'])
plt.legend(['Volume'])
plt.ylabel('Volume of stock')
plt.xlabel('Days')
plt.show()
### Noemalizing the training data 
trainset=data.loc[:,['open','high','low','volume','close']]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(trainset)
training_set_scaled.shape

### changing the dimension of data to fit it into an LSTM layer
def time_step_incorporation_train_set(time_step):
    x_train=[]
    y_train=[]
    for i in range(time_step,4971):
        x_train.append(training_set_scaled[i-time_step:i])
        y_train.append(training_set_scaled[i,0])
    x_train,y_train = np.array(x_train),np.array(y_train)
   
    return x_train,y_train
## Test set data
data2=pd.read_csv('/kaggle/input/bovespa-index-ibovespa-stocks-data/AZUL4_series.csv')
data2.head()
features = ['open','high','low','close','volume']
testset = data2[features]
test_set_scaled = sc.fit_transform(testset)
test_set_scaled.shape
def time_step_incorporation_test_set(time_step):
    x_test=[]
    y_test=[]
    for i in range(time_step,732):
        x_test.append(test_set_scaled[i-time_step:i])
        y_test.append(test_set_scaled[i,0])
    x_test,y_test = np.array(x_test),np.array(y_test)
    return x_test,y_test
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
X_train,Y_train=time_step_incorporation_train_set(100)
X_train.shape,Y_train.shape
### LSTM model ###
model = Sequential()
model.add(LSTM(units=70,return_sequences=True,input_shape=(X_train.shape[1], 5)))
model.add(Dropout(0.3))
model.add(LSTM(units=60,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=60,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=60))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=20,batch_size=32)

X_test,Y_test = time_step_incorporation_test_set(100)
y_prediction = model.predict(X_test)

sc.scale_

scale = 1/1.89573460e-02
y_prediction = y_prediction * scale
Y_test = Y_test * scale
plt.figure(num = 'Date',figsize = (10,5))
plt.plot(y_prediction,color = 'blue')
plt.plot(Y_test,color = 'red')
plt.xlabel('Days')
plt.ylabel('price (USD)')
plt.legend(['price_prediction','Actual_price'])
plt.show()