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
#import libraries
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_log_error

from datetime import datetime
from datetime import timedelta

from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import requests
import numpy as np
import pandas as pd
import io

BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
CONFIRMED = 'time_series_covid19_confirmed_global.csv'
DEATH = 'time_series_covid19_deaths_global.csv'
RECOVERED = 'time_series_covid19_recovered_global.csv'
CONFIRMED_US = 'time_series_covid19_confirmed_US.csv'
DEATH_US = 'time_series_covid19_deaths_US.csv'

def get_covid_data(subset = 'CONFIRMED'):
    """This function returns the latest available data subset of COVID-19. 
        The returned value is in pandas DataFrame type.
    Args:
        subset (:obj:`str`, optional): Any value out of 5 subsets of 'CONFIRMED',
        'DEATH', 'RECOVERED', 'CONFIRMED_US' and 'DEATH_US' is a valid input. If the value
        is not chosen or typed wrongly, CONFIRMED subet will be returned.
    """    

    if subset.upper() == 'DEATH':
        CSV_URL = BASE_URL + DEATH
    elif subset.upper() == 'RECOVERED':
        CSV_URL = BASE_URL + RECOVERED        
    elif subset.upper() == 'CONFIRMED_US':
        CSV_URL = BASE_URL + CONFIRMED_US        
    elif subset.upper() == 'DEATH_US':
        CSV_URL = BASE_URL + DEATH_US        
    else:
        CSV_URL = BASE_URL + CONFIRMED

    with requests.Session() as s:
        download = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        data = pd.read_csv(io.StringIO(decoded_content))

    return data
df_train_confirmed = get_covid_data(subset = 'CONFIRMED')
df_train_confirmed.head()
# We will fill the missing states with a value 'NoState'
df_train_confirmed=df_train_confirmed.fillna('NoState')
# changing the data type
df_train_confirmed.head()
#train=train.rename(columns={ConfirmedCases:'Confirmed','Country_Region':'Country/Region',
                    # 'Province_State':'Province/State','Date':'ObservationDate'})
#num_cols=['Confirmed']
#for col in num_cols:
#    temp=[int(i) for i in train[col]]
#    train[col]=temp 
#train.head(2)
#countries=['India','Italy','Spain']

#y=df_train_confirmed.loc[df_train_confirmed['Country/Region']=='Brazil'].iloc[0,4:]
#s = pd.DataFrame({'Brazil':y})
#for c in countries:    
    #pyplot.plot(range(y.shape[0]),y,'r--')
#    s[c] = df_train_confirmed.loc[df_train_confirmed['Country/Region']==c].iloc[0,4:]
#pyplot.plot(range(y.shape[0]),y,'g-')
#plt.plot(range(y.shape[0]), s)
#plt.legend(countries)

#s.tail(5)


df_usa1 = df_train_confirmed.loc[df_train_confirmed["Country/Region"]== "US"]
df_spain=df_train_confirmed.loc[df_train_confirmed["Country/Region"]== "Spain"]
df_india=df_train_confirmed.loc[df_train_confirmed["Country/Region"]== "India"]
df_brazil=df_train_confirmed.loc[df_train_confirmed["Country/Region"]== "Brazil"]
dates1=df_train_confirmed[4:]
temp_usa = df_usa1.melt(value_vars=dates1, var_name='Date', value_name='Confirmed')
temp_usa = temp_usa.groupby('Date')['Confirmed'].sum().reset_index()

pr_usa = pd.DataFrame(temp_usa)

pr_usa.columns = ['ds','y']
pr_usa.drop(pr_usa.tail(4).index,inplace=True)

#Spain
temp_spain = df_spain.melt(value_vars=dates1, var_name='Date', value_name='Confirmed')
temp_spain = temp_spain.groupby('Date')['Confirmed'].sum().reset_index()

pr_spain = pd.DataFrame(temp_spain)

pr_spain.columns = ['ds','y']
pr_spain.drop(pr_spain.tail(4).index,inplace=True)

#India
temp_india = df_india.melt(value_vars=dates1, var_name='Date', value_name='Confirmed')
temp_india = temp_india.groupby('Date')['Confirmed'].sum().reset_index()

pr_india = pd.DataFrame(temp_india)

pr_india.columns = ['ds','y']
pr_india.drop(pr_india.tail(4).index,inplace=True)

#Brazil
temp_brazil = df_brazil.melt(value_vars=dates1, var_name='Date', value_name='Confirmed')
temp_brazil = temp_brazil.groupby('Date')['Confirmed'].sum().reset_index()

pr_brazil = pd.DataFrame(temp_brazil)

pr_brazil.columns = ['ds','y']
pr_brazil.drop(pr_brazil.tail(4).index,inplace=True)

import fbprophet
#USA
m=fbprophet.Prophet()
m.fit(pr_usa)
future=m.make_future_dataframe(periods=10)
forecast_usa=m.predict(future)
forecast_usa.tail(11)

#Spain
m=fbprophet.Prophet()
m.fit(pr_spain)
future=m.make_future_dataframe(periods=10)
forecast_spain=m.predict(future)
forecast_spain.tail(11)

#India

m=fbprophet.Prophet()
m.fit(pr_india)
future=m.make_future_dataframe(periods=10)
forecast_india=m.predict(future)
forecast_india.tail(11)

#Brazil
m=fbprophet.Prophet()
m.fit(pr_brazil)
future=m.make_future_dataframe(periods=10)
forecast_brazil=m.predict(future)
forecast_brazil.tail(11)
cnfrm = forecast_usa.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm=cnfrm.tail(15)
cnfrm.columns = ['Date','Confirm_prophet_usa']
prophet_data_usa=cnfrm.tail(10)
prophet_data_usa
#Spain
cnfrm = forecast_spain.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm=cnfrm.tail(15)
cnfrm.columns = ['Date','Confirm_prophet_spain']
prophet_data_spain=cnfrm.tail(10)
prophet_data_spain
#India
cnfrm = forecast_india.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm=cnfrm.tail(15)
cnfrm.columns = ['Date','Confirm_prophet_india']
prophet_data_india=cnfrm.tail(10)
prophet_data_india
#Brazil
cnfrm = forecast_brazil.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm=cnfrm.tail(15)
cnfrm.columns = ['Date','Confirm_prophet_brazil']
prophet_data_brazil=cnfrm.tail(10)
prophet_data_brazil

df_usa1 = df_usa1.iloc[:, 5:]
df_spain= df_spain.iloc[:,5:]
df_india = df_india.iloc[:, 5:]
df_brazil= df_brazil.iloc[:,5:]

#Predicting future cases of USA using LSTM network
daily_cases = df_usa1.sum(axis=0)
daily_cases.index = pd.to_datetime(daily_cases.index)
daily_cases.tail()
#we take 20% of total data for validation and rest for testing i.e;(as of now 80 days first 64 days for testing and 16 days for validation)
test_data_size = 16

train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]

train_data.shape
#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(train_data, axis=1))

train_data = scaler.transform(np.expand_dims(train_data, axis=1))

test_data = scaler.transform(np.expand_dims(test_data, axis=1))

#Create a sequence
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

import torch
seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

import torch.nn as nn
class CoronaVirusPredictor(nn.Module):

  def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
    super(CoronaVirusPredictor, self).__init__()

    self.n_hidden = n_hidden
    self.seq_len = seq_len
    self.n_layers = n_layers

    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,
      dropout=0.5
    )

    self.linear = nn.Linear(in_features=n_hidden, out_features=1)

  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
    )

  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
      sequences.view(len(sequences), self.seq_len, -1),
      self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred
def train_model(
  model, 
  train_data, 
  train_labels, 
  test_data=None, 
  test_labels=None
):
  loss_fn = torch.nn.MSELoss(reduction='sum')

  optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 500   # if its time consuming then set it to 500 or 100
  train_hist = np.zeros(num_epochs)
  test_hist = np.zeros(num_epochs)

  for t in range(num_epochs):
    model.reset_hidden_state()

    y_pred = model(X_train)

    loss = loss_fn(y_pred.float(), y_train)

    if test_data is not None:
      with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred.float(), y_test)
      test_hist[t] = test_loss.item()

      if t % 100 == 0:  
        print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
    elif t % 100 == 0:
      print(f'Epoch {t} train loss: {loss.item()}')

    train_hist[t] = loss.item()
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()
  
  return model.eval(), train_hist, test_hist

#Training the model
model = CoronaVirusPredictor(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, train_hist, test_hist = train_model(
  model, 
  X_train, 
  y_train, 
  X_test, 
  y_test
)

with torch.no_grad():
  test_seq = X_test[:1]
  preds = []
  for _ in range(len(X_test)):
    y_test_pred = model(test_seq)
    pred = torch.flatten(y_test_pred).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
    
true_cases_usa = scaler.inverse_transform(
np.expand_dims(y_test.flatten().numpy(), axis=0)
).flatten()

predicted_cases_usa = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()
#Predicting future cases of spain using LSTM network
daily_cases = df_spain.sum(axis=0)
daily_cases.index = pd.to_datetime(daily_cases.index)
daily_cases.tail()
#we take 20% of total data for validation and rest for testing i.e;(as of now 80 days first 64 days for testing and 16 days for validation)
test_data_size = 16

train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]

train_data.shape
#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(train_data, axis=1))

train_data = scaler.transform(np.expand_dims(train_data, axis=1))

test_data = scaler.transform(np.expand_dims(test_data, axis=1))

#Create a sequence
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

import torch
seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

import torch.nn as nn
class CoronaVirusPredictor(nn.Module):

  def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
    super(CoronaVirusPredictor, self).__init__()

    self.n_hidden = n_hidden
    self.seq_len = seq_len
    self.n_layers = n_layers

    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,
      dropout=0.5
    )

    self.linear = nn.Linear(in_features=n_hidden, out_features=1)

  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
    )

  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
      sequences.view(len(sequences), self.seq_len, -1),
      self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred
def train_model(
  model, 
  train_data, 
  train_labels, 
  test_data=None, 
  test_labels=None
):
  loss_fn = torch.nn.MSELoss(reduction='sum')

  optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 500   # if its time consuming then set it to 500 or 100
  train_hist = np.zeros(num_epochs)
  test_hist = np.zeros(num_epochs)

  for t in range(num_epochs):
    model.reset_hidden_state()

    y_pred = model(X_train)

    loss = loss_fn(y_pred.float(), y_train)

    if test_data is not None:
      with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred.float(), y_test)
      test_hist[t] = test_loss.item()

      if t % 100 == 0:  
        print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
    elif t % 100 == 0:
      print(f'Epoch {t} train loss: {loss.item()}')

    train_hist[t] = loss.item()
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()
  
  return model.eval(), train_hist, test_hist

#Training the model
model = CoronaVirusPredictor(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, train_hist, test_hist = train_model(
  model, 
  X_train, 
  y_train, 
  X_test, 
  y_test
)

with torch.no_grad():
  test_seq = X_test[:1]
  preds = []
  for _ in range(len(X_test)):
    y_test_pred = model(test_seq)
    pred = torch.flatten(y_test_pred).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
    
true_cases_spain = scaler.inverse_transform(
np.expand_dims(y_test.flatten().numpy(), axis=0)
).flatten()

predicted_cases_spain = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()
#Predicting future cases of India using LSTM network
daily_cases = df_india.sum(axis=0)
daily_cases.index = pd.to_datetime(daily_cases.index)
daily_cases.tail()
#we take 20% of total data for validation and rest for testing i.e;(as of now 80 days first 64 days for testing and 16 days for validation)
test_data_size = 16

train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]

train_data.shape
#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(train_data, axis=1))

train_data = scaler.transform(np.expand_dims(train_data, axis=1))

test_data = scaler.transform(np.expand_dims(test_data, axis=1))

#Create a sequence
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

import torch
seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

import torch.nn as nn
class CoronaVirusPredictor(nn.Module):

  def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
    super(CoronaVirusPredictor, self).__init__()

    self.n_hidden = n_hidden
    self.seq_len = seq_len
    self.n_layers = n_layers

    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,
      dropout=0.5
    )

    self.linear = nn.Linear(in_features=n_hidden, out_features=1)

  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
    )

  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
      sequences.view(len(sequences), self.seq_len, -1),
      self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred
def train_model(
  model, 
  train_data, 
  train_labels, 
  test_data=None, 
  test_labels=None
):
  loss_fn = torch.nn.MSELoss(reduction='sum')

  optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 500   # if its time consuming then set it to 500 or 100
  train_hist = np.zeros(num_epochs)
  test_hist = np.zeros(num_epochs)

  for t in range(num_epochs):
    model.reset_hidden_state()

    y_pred = model(X_train)

    loss = loss_fn(y_pred.float(), y_train)

    if test_data is not None:
      with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred.float(), y_test)
      test_hist[t] = test_loss.item()

      if t % 100 == 0:  
        print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
    elif t % 100 == 0:
      print(f'Epoch {t} train loss: {loss.item()}')

    train_hist[t] = loss.item()
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()
  
  return model.eval(), train_hist, test_hist

#Training the model
model = CoronaVirusPredictor(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, train_hist, test_hist = train_model(
  model, 
  X_train, 
  y_train, 
  X_test, 
  y_test
)

with torch.no_grad():
  test_seq = X_test[:1]
  preds = []
  for _ in range(len(X_test)):
    y_test_pred = model(test_seq)
    pred = torch.flatten(y_test_pred).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
    
true_cases_india = scaler.inverse_transform(
np.expand_dims(y_test.flatten().numpy(), axis=0)
).flatten()

predicted_cases_india = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()
#Predicting future cases of Brazil using LSTM network
daily_cases = df_brazil.sum(axis=0)
daily_cases.index = pd.to_datetime(daily_cases.index)
daily_cases.tail()
#we take 20% of total data for validation and rest for testing i.e;(as of now 80 days first 64 days for testing and 16 days for validation)
test_data_size = 16

train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]

train_data.shape
#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(train_data, axis=1))

train_data = scaler.transform(np.expand_dims(train_data, axis=1))

test_data = scaler.transform(np.expand_dims(test_data, axis=1))

#Create a sequence
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

import torch
seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

import torch.nn as nn
class CoronaVirusPredictor(nn.Module):

  def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
    super(CoronaVirusPredictor, self).__init__()

    self.n_hidden = n_hidden
    self.seq_len = seq_len
    self.n_layers = n_layers

    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,
      dropout=0.5
    )

    self.linear = nn.Linear(in_features=n_hidden, out_features=1)

  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
    )

  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
      sequences.view(len(sequences), self.seq_len, -1),
      self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred
def train_model(
  model, 
  train_data, 
  train_labels, 
  test_data=None, 
  test_labels=None
):
  loss_fn = torch.nn.MSELoss(reduction='sum')

  optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 500   # if its time consuming then set it to 500 or 100
  train_hist = np.zeros(num_epochs)
  test_hist = np.zeros(num_epochs)

  for t in range(num_epochs):
    model.reset_hidden_state()

    y_pred = model(X_train)

    loss = loss_fn(y_pred.float(), y_train)

    if test_data is not None:
      with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred.float(), y_test)
      test_hist[t] = test_loss.item()

      if t % 100 == 0:  
        print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
    elif t % 100 == 0:
      print(f'Epoch {t} train loss: {loss.item()}')

    train_hist[t] = loss.item()
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()
  
  return model.eval(), train_hist, test_hist

#Training the model
model = CoronaVirusPredictor(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, train_hist, test_hist = train_model(
  model, 
  X_train, 
  y_train, 
  X_test, 
  y_test
)

with torch.no_grad():
  test_seq = X_test[:1]
  preds = []
  for _ in range(len(X_test)):
    y_test_pred = model(test_seq)
    pred = torch.flatten(y_test_pred).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
    
true_cases_brazil = scaler.inverse_transform(
np.expand_dims(y_test.flatten().numpy(), axis=0)
).flatten()

predicted_cases_brazil = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()
prophet_data_usa.insert(2,'LSTM_usa',predicted_cases_usa)
prophet_data_spain.insert(2,'LSTM_spain',predicted_cases_spain)
prophet_data_india.insert(2,'LSTM_india',predicted_cases_india)
prophet_data_brazil.insert(2,'LSTM_brazil',predicted_cases_brazil)
USA_data=pd.DataFrame(prophet_data_usa)
USA_data['FutureDates']=USA_data['Date'].apply(lambda x: x.strftime('%d%m%Y'))

Spain_data=pd.DataFrame(prophet_data_spain)
Spain_data['FutureDates']=Spain_data['Date'].apply(lambda x: x.strftime('%d%m%Y'))

India_data=pd.DataFrame(prophet_data_india)
India_data['FutureDates']=India_data['Date'].apply(lambda x: x.strftime('%d%m%Y'))

Brazil_data=pd.DataFrame(prophet_data_brazil)
Brazil_data['FutureDates']=Brazil_data['Date'].apply(lambda x: x.strftime('%d%m%Y'))

fig, axes = plt.subplots(nrows=2, ncols=2)

USA_data.plot(kind='line',x='FutureDates',y='Confirm_prophet_usa',ax=axes[0,0],figsize=(10,10))
USA_data.plot(kind='line',x= 'FutureDates',y='LSTM_usa', color='red', ax=axes[0,0])
axes[0, 0].set_title('USA Future Confirmed cases')

Spain_data.plot(kind='line',x='FutureDates',y='Confirm_prophet_spain',ax=axes[0,1],figsize=(10,10))
Spain_data.plot(kind='line',x= 'FutureDates',y='LSTM_spain', color='red', ax=axes[0,1])
axes[0, 1].set_title('Spain Future Confirmed cases')

India_data.plot(kind='line',x='FutureDates',y='Confirm_prophet_india',ax=axes[1,0],figsize=(10,10))
India_data.plot(kind='line',x= 'FutureDates',y='LSTM_india', color='red', ax=axes[1,0])
axes[1, 0].set_title('India Future Confirmed cases')

Brazil_data.plot(kind='line',x='FutureDates',y='Confirm_prophet_brazil',ax=axes[1,1],figsize=(10,10))
Brazil_data.plot(kind='line',x= 'FutureDates',y='LSTM_brazil', color='red', ax=axes[1,1])
axes[1, 1].set_title('Brazil Future Confirmed cases')

plt.show()