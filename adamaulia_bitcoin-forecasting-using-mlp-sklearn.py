import pandas as pd 
%matplotlib inline 
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('../input/bitcoin_price.csv',parse_dates=['Date'])
data.head()
data.describe()
df_close = data[['Date','Close']]
df_close.head(3)
df_close = df_close.set_index('Date')
print ('before sort')
print (df_close.head(3))

df_close.sort_index(inplace=True)

print ('after sort')
print (df_close.head(3))
ax = df_close.plot()
ax.set_xlabel('date')
ax.set_ylabel('price in usd')
#scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_close['scale'] = scaler.fit_transform(df_close['Close'].values.reshape(-1,1))
df_close['scale'].head()
ax = df_close.plot(y='scale')
ax.set_xlabel('date')
ax.set_ylabel('price in usd')
#create historical data
def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
x, y = create_dataset(df_close['scale'])
print (x.shape)
print (y.shape)
plt.plot(y)
plt.title('forcasting target')
size = int(len(x) * 0.70)

x_train, x_test = x[0:size], x[size:len(x)]
y_train, y_test = y[0:size], y[size:len(x)]


print ('x_train',x_train.shape, x_train[:5],y_train[:5])
print ('x_test',x_test.shape, x_test[:5],y_test[:5])

plt.plot(y_train)
plt.title('training forecasting target')
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(activation = 'tanh',solver='adam')
clf.fit(x_train,y_train)

train_mse = clf.predict(x_train)
test_mse = clf.predict(x_test)
print ('MSE training', mean_squared_error(train_mse,y_train))
print ('MSE testing', mean_squared_error(test_mse,y_test))
train_pred = clf.predict(x_train)
test_pred = clf.predict(x_test)

plt.plot(train_pred,label='training result')
plt.plot(y_train,color='red', label='prediction target')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('training result')
plt.plot(test_pred,label='testing result')
plt.plot(y_test,color='red',label='prediction target')
#plt.legend('test,prediction', ncol=2, loc='upper left');
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('testing result')
plt.plot(scaler.inverse_transform(train_pred.reshape(-1,1)),label='training result')
plt.plot(scaler.inverse_transform(y_train.reshape(-1,1)),color='red', label='prediction target')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('training result with real number')
plt.plot(scaler.inverse_transform(test_pred.reshape(-1,1)),label='testing result')
plt.plot(scaler.inverse_transform(y_test.reshape(-1,1)),color='red', label='prediction target')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('testing result with real number')