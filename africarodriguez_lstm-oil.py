import numpy as np 
import pandas as pd 
import datetime
from pylab import rcParams
import matplotlib.pyplot as plt
import warnings
import itertools
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import math
from sklearn.preprocessing import MinMaxScaler
crude = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv')

crude.head(3)
crude.tail(3)
crude.info()
crude['date'] = pd.to_datetime(crude.Date)
crude.info()
crude.set_index('date')
crude.info()
crude=crude.drop(['Date'], axis=1)
crude.set_index(['date'], inplace=True)
y = crude['Price'].resample('MS').mean()
y.plot(figsize=(15, 6))
plt.show()
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()
data=pd.DataFrame()
data=crude
sc = MinMaxScaler(feature_range = (0, 1))
df = sc.fit_transform(data)

train_size = int(len(df) * 0.70)
test_size = len(df) - train_size
train, test = df[0:train_size, :], df[train_size:len(df), :]
def create_data_set(_data_set, _look_back=1):
    data_x, data_y = [], []
    for i in range(len(_data_set) - _look_back - 1):
        a = _data_set[i:(i + _look_back), 0]
        data_x.append(a)
        data_y.append(_data_set[i + _look_back, 0])
    return np.array(data_x), np.array(data_y)
look_back =75
X_train,Y_train,X_test,Ytest = [],[],[],[]
X_train,Y_train=create_data_set(train,look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
X_test,Y_test=create_data_set(test,look_back)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
X_train.shape
Y_train.shape
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train,Y_train)
reg.score(X_train,Y_train)
y_test=reg.predict(X_test)
df1=pd.DataFrame(Y_test, columns=['True'])
df1['Prediction']=y_test
df1['Diference']=df1['True']-df1['Prediction']

df1
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1['True'], df1['Prediction'],  multioutput='raw_values')
mse
data.info()
sc = MinMaxScaler(feature_range = (0, 1))
df = sc.fit_transform(data)
df.shape
train_size = int(len(df) * 0.70)
test_size = len(df) - train_size
train, test = df[0:train_size, :], df[train_size:len(df), :]
def create_data_set(_data_set, _look_back=1):
    data_x, data_y = [], []
    for i in range(len(_data_set) - _look_back - 1):
        a = _data_set[i:(i + _look_back), 0]
        data_x.append(a)
        data_y.append(_data_set[i + _look_back, 0])
    return np.array(data_x), np.array(data_y)
look_back = 75
X_train,Y_train,X_test,Ytest = [],[],[],[]
X_train,Y_train=create_data_set(train,look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test,Y_test=create_data_set(test,look_back)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
regressor = Sequential()

regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units = 60))
regressor.add(Dropout(0.1))

regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5)
history =regressor.fit(X_train, Y_train, epochs = 20, batch_size = 15,validation_data=(X_test, Y_test), callbacks=[reduce_lr],shuffle=False)

train_predict = regressor.predict(X_train)
test_predict = regressor.predict(X_test)
train_predict = sc.inverse_transform(train_predict)
Y_train = sc.inverse_transform([Y_train])
test_predict = sc.inverse_transform(test_predict)
Y_test = sc.inverse_transform([Y_test])
print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();
aa=[x for x in range(180)]
plt.figure(figsize=(8,4))
plt.plot(aa, Y_test[0][:180], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:180], 'r', label="prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()
test_predict.shape
df=pd.DataFrame()
df=data
ultimosDias = df['2020-01-01':'2020-07-04']
ultimosDias

ultimosDias.shape
sc = MinMaxScaler(feature_range = (0, 1))
df_p = sc.fit_transform(ultimosDias)
testX, testY = create_data_set(df_p, _look_back=75)
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
testX
test_predict = regressor.predict(testX, batch_size=1)
test_predict.shape
test_predict = sc.inverse_transform(test_predict)
test_predict
Date1 = pd.date_range('2020-07-16', periods=740, freq='D')
columns = ['Date','Price']    
Test2 = pd.DataFrame(columns=columns)
Test2['Price'] = pd.to_numeric(Test2['Price'])
Test2["Date"] = pd.to_datetime(Date1)
Test2 = Test2.fillna(0)
Test2.set_index(['Date'], inplace=True)

from random import randrange
Test2.head(3)
Test2['Price']=np.random.uniform(39,41, Test2.shape[0])
Test2.head(3)
test=sc.fit_transform(Test2)
test.shape
test.shape[0]
def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)
testX, testY = create_dataset(test, look_back=75)
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
test_predict = regressor.predict(testX, batch_size=15)
test_predict = sc.inverse_transform(test_predict)
df_f=pd.DataFrame(Test2['Price'][:37], columns=['Price'])
df_f['Price']=test_predict[:37]
df_f.head(3)
df_f.shape
df_f.to_csv('sampleSubmission.csv', index=True)
y=df_f['Price']
y.plot(figsize=(15, 6))
plt.show()