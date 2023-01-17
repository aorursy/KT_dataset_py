import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from tqdm import tqdm
sns.set_style("darkgrid")


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, Input


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df= pd.read_csv('/kaggle/input/nyse/prices-split-adjusted.csv')
df.head()
amz= df[df.symbol == 'AMZN']
amz.shape
print('Select region of interest for deeper understanding ')
px.line(amz, x="date", y=["open", "close","low", "high"], title='Opening stock prices of Amazon.com, Inc.')
fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4])

fig.add_trace(go.Scatter(x= amz.date, y=amz.open.diff(), name='l1'),
              row=1, col=1)

fig.add_trace(go.Histogram(x=amz['open'].diff(), name='h1', histnorm='probability density'),
              row=1, col=2)
fig.update_layout( height=550, width=1130, title_text="Consecutive difference between opening stock price of AMAZON shares")

fig.update_xaxes(title_text="Time", row=1, col=1);   fig.update_xaxes(title_text="Value", row=1, col=2)
fig.update_yaxes(title_text="Value", row=1, col=1);   fig.update_yaxes(title_text="Prob. Density", row=1, col=2)

fig.show()
f, axes= plt.subplots(2,2, figsize=(20,14))
sns.regplot(x=amz.open, y=amz.close, color="g", ax=axes[0][0])
sns.regplot(x=amz.open, y=amz.volume, ax=axes[0][1])
sns.regplot(x=amz.low, y=amz.high, color="b", ax=axes[1][0])
sns.regplot(x=amz.volume, y=amz.close, color="g", ax=axes[1][1])
f, axes= plt.subplots(1,2, figsize=(20,6))
sns.regplot(x=amz.open.diff(), y=amz.close.diff(), color="g", ax=axes[0])
sns.regplot(x=amz.low.diff(), y=amz.high.diff(), color="b", ax=axes[1])
plt.suptitle('Consecutive variation correlations', size=16)
corr= amz.corr()
plt.figure(figsize=(8,5))
sns.heatmap(corr, annot=True, cmap="Greens_r",linewidth = 3, linecolor = "white")
def split_data(X,Y):
    return X[:-40], Y[:-40], X[-40:], Y[-40:]
def data_for_linear(df, timesteps= 40):
    x=amz.open.values
    x=x.reshape(-1,1)
    #Normalization
    
    scaler = MinMaxScaler()
    x_noml=scaler.fit_transform(x)
    X=[]; Y=[]
    for i in tqdm(range(x.shape[0]- timesteps)):
        X.append(x_noml[i:i+timesteps])
        Y.append(x_noml[i+timesteps])
    X=np.array(X); Y= np.array(Y)
    X= np.reshape(X, (-1,timesteps))
    print('Input shape:{}, Output shape:{}'.format(X.shape, Y.shape))
    return X,Y, scaler
#Loading & data splitting
# 2D Input 
x_linear, ylinear, lin_scaler= data_for_linear(amz)
xtrain, ytrain, xtest, ytest= split_data(x_linear, ylinear)
liner_model= Sequential()
liner_model.add(Dense(128, activation=None, input_shape=(40,)))
liner_model.add(Dense(228, activation=None))
liner_model.add(Dense(64, activation=None))
liner_model.add(Dense(1, activation=None))

liner_model.compile(optimizer='adam', loss='mse')
liner_model.summary()
his= liner_model.fit(xtrain, ytrain, epochs=40)
fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5])

fig.add_trace(go.Scatter(x= amz.date[40:1682].values, y= lin_scaler.inverse_transform(ytrain).reshape(-1), name='Traning data'),
              row=1, col=1)

fig.add_trace(go.Scatter(x= amz.date[40:1682].values, y= lin_scaler.inverse_transform(liner_model.predict(xtrain)).reshape(-1), name='Prediction'),
              row=1, col=1)

fig.add_trace(go.Scatter(x= amz.date[1682+40:].values, y= lin_scaler.inverse_transform(ytest).reshape(-1), name='Testing data'),
              row=1, col=2)

fig.add_trace(go.Scatter(x= amz.date[1682+40:].values, y= lin_scaler.inverse_transform(liner_model.predict(xtest)).reshape(-1), name='Test prediction'),
              row=1, col=2)

fig.update_layout( height=550, width=1200, title_text="Linear Model performance")

fig.update_xaxes(title_text="Date", row=1, col=1);   fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_yaxes(title_text="Opening stock price", row=1, col=1);   fig.update_yaxes(title_text="Opening stock price", row=1, col=2)

fig.show()
#-------------- True Forecasting/predicting future sales--------------------#
ypred=[]
for i in range(xtest.shape[0]-1):
    p= liner_model.predict(xtest[i].reshape(1,40))
    ypred.append(p)
    xtest[i+1, -1]= p
ypred.append(liner_model.predict(xtest[i+1].reshape(1,40)))
#-------------- Note it!------------------------------------------------------#
    

f, ax= plt.subplots(1, 2, figsize=(30,8))
ax[0].plot(lin_scaler.inverse_transform(ytrain))
ax[0].plot(lin_scaler.inverse_transform(liner_model.predict(xtrain)))
ax[0].set_title('Traning set, mse:{}'.format(mean_absolute_error(ytrain, liner_model.predict(xtrain))*100))
ax[0].set_ylabel('Open-stock-price')
ax[0].set_xlabel('Time')

ax[1].plot(lin_scaler.inverse_transform(ytest))
ax[1].plot(lin_scaler.inverse_transform(np.array(ypred).reshape(-1,1)))
ax[1].set_title('Testing-set, mse:{}'.format(mean_absolute_error(ytest, np.array(ypred).reshape(-1,1))*100))
ax[1].set_ylabel('Open-stock-price')
ax[1].set_xlabel('Time')
plt.show()
def data_for_RNNs(df, timesteps= 40):
    x=amz.open.values
    x=x.reshape(-1,1)
    
    #Normalization
    scaler = MinMaxScaler()
    x_noml=scaler.fit_transform(x)
    
    X=[]; Y=[]
    for i in tqdm(range(x.shape[0]- timesteps)):
        X.append(x_noml[i:i+timesteps])
        Y.append(x_noml[i+timesteps])
    X=np.array(X); Y= np.array(Y)
    print('Input shape:{}, Output shape:{}'.format(X.shape, Y.shape))
    return X,Y, scaler
# 3D Input 
x_rnn, y_rnn, rnn_scaler= data_for_RNNs(amz)
xtrain, ytrain, xtest, ytest= split_data(x_rnn, y_rnn)
rnn_model= Sequential()
rnn_model.add(SimpleRNN(128, activation='tanh', input_shape=(40,1), return_sequences=True))
rnn_model.add(SimpleRNN(228,activation='tanh', return_sequences=True))
rnn_model.add(Dropout(0.3))
rnn_model.add(SimpleRNN(64, activation='tanh', return_sequences=False))
rnn_model.add(Dense(1, activation=None))

rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.summary()
his= rnn_model.fit(xtrain, ytrain, epochs=40)
fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5])

fig.add_trace(go.Scatter(x= amz.date[40:1682].values, y= rnn_scaler.inverse_transform(ytrain).reshape(-1), name='Traning data'),
              row=1, col=1)

fig.add_trace(go.Scatter(x= amz.date[40:1682].values, y= rnn_scaler.inverse_transform(rnn_model.predict(xtrain)).reshape(-1), name='Prediction'),
              row=1, col=1)

fig.add_trace(go.Scatter(x= amz.date[1682+40:].values, y= rnn_scaler.inverse_transform(ytest).reshape(-1), name='Testing data'),
              row=1, col=2)

fig.add_trace(go.Scatter(x= amz.date[1682+40:].values, y= rnn_scaler.inverse_transform(rnn_model.predict(xtest)).reshape(-1), name='Test prediction'),
              row=1, col=2)

fig.update_layout( height=550, width=1200, title_text="Simple RNN Model performance")

fig.update_xaxes(title_text="Date", row=1, col=1);   fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_yaxes(title_text="Opening stock price", row=1, col=1);   fig.update_yaxes(title_text="Opening stock price", row=1, col=2)

fig.show()
#-------------- True Forecasting/predicting of future data--------------------#

# Grab value predicted by model on 1st sample of test data.
# Replace last(timestep) value from 2nd sample with predicted value.
# Predict.....Grab.....Replace....Predict.... 


ypred=[]
for i in range(xtest.shape[0]-1):
    p= rnn_model.predict(xtest[i].reshape(1,40, 1))
    ypred.append(p)
    xtest[i+1, -1]= p
ypred.append(rnn_model.predict(xtest[i+1].reshape(1,40, 1)))
#-------------- Note it!------------------------------------------------------#
    

f, ax= plt.subplots(1, 2, figsize=(30,8))
ax[0].plot(rnn_scaler.inverse_transform(ytrain))
ax[0].plot(rnn_scaler.inverse_transform(rnn_model.predict(xtrain)))
ax[0].set_title('Traning set, mse:{}'.format(mean_absolute_error(ytrain, rnn_model.predict(xtrain))*100))
ax[0].set_ylabel('Open-stock-price')
ax[0].set_xlabel('Time')

ax[1].plot(rnn_scaler.inverse_transform(ytest))
ax[1].plot(rnn_scaler.inverse_transform(np.array(ypred).reshape(-1,1)))
ax[1].set_title('Testing-set, mse:{}'.format(mean_absolute_error(ytest, np.array(ypred).reshape(-1,1))*100))
ax[1].set_ylabel('Open-stock-price')
ax[1].set_xlabel('Time')
plt.show()
# 3D Input 
x_lstm, y_lstm, lstm_scaler= data_for_RNNs(amz)
xtrain, ytrain, xtest, ytest= split_data(x_lstm, y_lstm)
lstm_model= Sequential()
lstm_model.add(LSTM(128, activation='tanh', input_shape=(40,1), return_sequences=True))
lstm_model.add(LSTM(228,activation='tanh', return_sequences=True))
lstm_model.add(Dropout(0.3))
lstm_model.add(LSTM(64, activation='tanh', return_sequences=False))
lstm_model.add(Dense(1, activation=None))

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()
his= lstm_model.fit(xtrain, ytrain, epochs=40)
fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5])

fig.add_trace(go.Scatter(x= amz.date[40:1682].values, y= lstm_scaler.inverse_transform(ytrain).reshape(-1), name='Traning data'),
              row=1, col=1)

fig.add_trace(go.Scatter(x= amz.date[40:1682].values, y= lstm_scaler.inverse_transform(lstm_model.predict(xtrain)).reshape(-1), name='Prediction'),
              row=1, col=1)

fig.add_trace(go.Scatter(x= amz.date[1682+40:].values, y= lstm_scaler.inverse_transform(ytest).reshape(-1), name='Testing data'),
              row=1, col=2)

fig.add_trace(go.Scatter(x= amz.date[1682+40:].values, y= lstm_scaler.inverse_transform(lstm_model.predict(xtest)).reshape(-1), name='Test prediction'),
              row=1, col=2)

fig.update_layout( height=550, width=1200, title_text="LSTM RNN Model performance")

fig.update_xaxes(title_text="Date", row=1, col=1);   fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_yaxes(title_text="Opening stock price", row=1, col=1);   fig.update_yaxes(title_text="Opening stock price", row=1, col=2)

fig.show()
#-------------- True Forecasting/predicting of future data--------------------#

# Grab value predicted by model on 1st sample of test data.
# Replace last(timestep) value from 2nd sample with predicted value.
# Predict.....Grab.....Replace....Predict.... 

ypred=[]
for i in range(xtest.shape[0]-1):
    p= lstm_model.predict(xtest[i].reshape(1,40, 1))
    ypred.append(p)
    xtest[i+1, -1]= p
ypred.append(lstm_model.predict(xtest[i+1].reshape(1,40, 1)))
#-------------- Note it!------------------------------------------------------#
    

f, ax= plt.subplots(1, 2, figsize=(30,8))
ax[0].plot(lstm_scaler.inverse_transform(ytrain))
ax[0].plot(lstm_scaler.inverse_transform(lstm_model.predict(xtrain)))
ax[0].set_title('Traning set, mse:{}'.format(mean_absolute_error(ytrain, lstm_model.predict(xtrain))*100))
ax[0].set_ylabel('Open-stock-price')
ax[0].set_xlabel('Time')

ax[1].plot(lstm_scaler.inverse_transform(ytest))
ax[1].plot(lstm_scaler.inverse_transform(np.array(ypred).reshape(-1,1)))
ax[1].set_title('Testing-set, mse:{}'.format(mean_absolute_error(ytest, np.array(ypred).reshape(-1,1))*100))
ax[1].set_ylabel('Open-stock-price')
ax[1].set_xlabel('Time')
plt.show()
