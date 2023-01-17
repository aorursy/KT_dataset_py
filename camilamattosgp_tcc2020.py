# importar as bibliotecas necessárias
import pandas as pd
from pandas_datareader import data as web
import plotly.graph_objects as go


import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dense


from keras.models import load_model



# criar um DataFrame vazio
df = pd.DataFrame()

look_back = 40
forward_days = 10
num_periods = 20
#open the csv, chose company_N, where N = {A, B, C or D}
df = pd.read_csv('../input/tcc-cotacao/price3.csv')
#df = pd.read_csv('/content/coffee-prices-historical-chart-data.csv')

#set date as index
df['date'] = pd.to_datetime(df['date'])
print(df['date'].dtype)
#df = df.loc[df['date'] >= '01-01-2003']
#df = df.loc[df['date'] <= '31-12-2013']
df.set_index('date', inplace=True)
#keep only the 'Close' column
df = df['value']
df.head()
len(df)
import matplotlib.pyplot as plt
plt.figure(figsize = (15,10))
plt.plot(df, label='Café Arábica')
plt.legend(loc='best')
plt.show()
array = df.values.reshape(df.shape[0],1)
array[:5]
from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
array = scl.fit_transform(array)
array[:5]
#split in Train and Test

division = len(array) - num_periods*forward_days

array_test = array[division-look_back:]
array_train = array[:division]
#Get the data and splits in input X and output Y, by spliting in `n` past days as input X 
#and `m` coming days as Y.
def processData(data, look_back, forward_days,jump=1):
    X,Y = [],[]
    for i in range(0,len(data) -look_back -forward_days +1, jump):
        X.append(data[i:(i+look_back)])
        Y.append(data[(i+look_back):(i+look_back+forward_days)])
    return np.array(X),np.array(Y)

X_test,y_test = processData(array_test,look_back,forward_days,forward_days)
y_test = np.array([list(a.ravel()) for a in y_test])

X,y = processData(array_train,look_back,forward_days)
y = np.array([list(a.ravel()) for a in y])
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape)
print(X_validate.shape)
print(X_test.shape)
print(y_train.shape)
print(y_validate.shape)
print(y_test.shape)
NUM_NEURONS_FirstLayer = 50
NUM_NEURONS_SecondLayer = 30
EPOCHS = 50

#Build the model
model = Sequential()
model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(look_back,1), return_sequences=True))
model.add(LSTM(NUM_NEURONS_SecondLayer,input_shape=(NUM_NEURONS_FirstLayer,1)))
model.add(Dense(forward_days))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train,y_train,epochs=EPOCHS,validation_data=(X_validate,y_validate),shuffle=True,batch_size=2, verbose=2)

plt.figure(figsize = (15,10))

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='best')
plt.show()

Xt = model.predict(X_test)
plt.figure(figsize = (15,10))

for i in range(0,len(Xt)):
    plt.plot([x + i*forward_days for x in range(len(Xt[i]))], scl.inverse_transform(Xt[i].reshape(-1,1)), color='r')
    
plt.plot(0, scl.inverse_transform(Xt[i].reshape(-1,1))[0], color='r', label='Prediction') #only to place the label
    
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label='Target')
plt.legend(loc='best')
plt.show()

