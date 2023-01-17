import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from fbprophet import Prophet
df = pd.read_json('../input/prices_2017.json')
df=df.set_index('date')
df.head()
df=df.drop('trading_code',axis=1)  
df.head()
%matplotlib inline
data2=df.reset_index()

data2.head()
data3 = data2.rename(columns={'date': 'ds'})

data3 = data3.rename(columns={'closing_price': 'y'})
data3.head()
data_for_pr1=data3[['ds','y']]
data_for_pr2=data3
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,20)
data_for_pr1['y'].plot()
data2['opening_price'].plot()
m=Prophet()
m.fit(data_for_pr1)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast.head()
forcast1=forecast[['ds', 'yhat']]
fig2 = m.plot_components(forecast)
!pip install tensorflow-gpu
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
df = pd.read_json('../input/prices_2017.json')
df=df.set_index('date')
df.head()
df=df.drop('trading_code',axis=1)  
corr = df.corr()
import seaborn as sns
sns.heatmap(corr)
df.isnull().sum()
df.corr()[['closing_price']]

df.corr()[['closing_price']].plot(kind='bar')
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
X = df.drop('closing_price',axis=True)

y = df[['closing_price']]


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2)

result = y_test

scaler = MinMaxScaler()

scaler.fit(y_train)

y_train =scaler.transform(y_train) # we transform the y so after predict we have to inverse transeform it

scaler.fit(y_test)

y_test =scaler.transform(y_test) # we transform the y so after predict we have to inverse transeform it



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers import Flatten
##have to convert to 3 dim for feeding RNN

x_train = np.array(x_train)

x_test = np.array(x_test)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))



print (x_train.shape)

print (x_test.shape)

print (y_train.shape)

print (y_test.shape)
def RNNMODEL():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1)) #we want single feature output which is df['Close']

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

model = RNNMODEL()
model.fit(x_train,y_train,epochs = 10)
y_pred = model.predict(x_test)
y_pred
## we have to inverse transform it cause we transform the x_test before

output = scaler.inverse_transform(y_pred)
output
real_output = []

for item in output:

    real_output.append((item[0]))
result['predited value'] = np.array(real_output)
result.head()

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,20)

result.plot()