import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

from sklearn import preprocessing 

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline



import os

print(os.listdir("../input"))

train = pd.read_csv('../input/air-passengers/AirPassengers.csv')
train.head()
train.shape
train['Month'].value_counts()
train['Month'] = train['Month'].astype('datetime64[ns]')
train = train.sort_values(by=['Month'])
train.head()
train['#Passengers'].hist()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"passenger":train['#Passengers'], "log(passenger + 1)":np.log1p(train['#Passengers'])})

prices.hist()
plt.style.use('fivethirtyeight')



plt.plot(train['Month'], train['#Passengers'], color='black')
train['index'] = train['#Passengers']

for i in range(train.shape[0]):

    train.at[i,'index'] = i;
train_df = train.loc[train['Month'] <= '1958-12-01']

valid_df = train.loc[train['Month'] > '1958-12-01']
print(train_df.shape)

print(valid_df.shape)
train_df.head()
def SMAPE(y_true, y_pred):

    error = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))

    error.replace([np.inf, -np.inf], np.nan, inplace=True)

    error.dropna(inplace=True)

    return np.mean(error)*100
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(np.array(train_df['index']).reshape(-1,1), train_df['#Passengers'])
plt.style.use('fivethirtyeight')



plt.scatter(train_df['index'], train_df['#Passengers'], color='black')

plt.plot(train_df['index'], model.predict(np.array(train_df['index']).reshape(-1,1)), color = 'blue')

plt.gca().set_title("Gradient Descent Linear Regressor")
print('SMAPE number passenger : ', SMAPE(train_df['#Passengers'], model.predict(np.array(train_df['index']).reshape(-1,1))))
plt.style.use('fivethirtyeight')



plt.scatter(valid_df['index'], valid_df['#Passengers'], color='black')

plt.plot(valid_df['index'], model.predict(np.array(valid_df['index']).reshape(-1,1)), color = 'blue')

plt.gca().set_title("Gradient Descent Linear Regressor")
print('SMAPE number passenger : ', SMAPE(valid_df['#Passengers'], model.predict(np.array(valid_df['index']).reshape(-1,1))))