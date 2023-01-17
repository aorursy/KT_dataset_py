import numpy as np

import pandas as pd
location = '../input/AirQualityUCI_req.csv'

target = 'NO2(GT)'
def get_data(location):

    df = pd.read_csv(location)

    

    df=df.set_index('Date')

    return df
df=get_data(location)

df.head()
df.corr()[['NO2(GT)']]
df.head()

df = df[['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)','PT08.S2(NMHC)', 'NOx(GT)','NO2(GT)','PT08.S5(O3)']]

df.head()
X = df.drop(target,axis=1)

y = df[[target]]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y)

value = y_test
print (x_train.shape)

print (x_test.shape)

print (y_train.shape)

print (y_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.fit_transform(x_train)

scaler.fit(x_test)

x_test = scaler.fit_transform(x_test)

scaler.fit(y_train)

y_train = scaler.fit_transform(y_train)

scaler.fit(y_test)

y_test = scaler.fit_transform(y_test)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
print (x_train.shape)

print (x_test.shape)

print (y_train.shape)

print (y_test.shape)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers import Flatten

def RNN_MODEL():

    # have to convert to 3D for feeding the data

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error',metrics=['accuracy'])

    return regressor



regressor = RNN_MODEL()
regressor.fit(x_train,y_train,epochs = 100)
regressor.evaluate(x_test,y_test)

y_pred = regressor.predict(x_test)
y_pred
output = scaler.inverse_transform(y_pred)

real_output = []

for item in output:

    real_output.append((item[0]))

real_output
value['predicted'] = np.array(real_output)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (100,100)
value.plot()