#import packages

import pandas as pd

import numpy as np



#to plot within notebook

import matplotlib.pyplot as plt

%matplotlib inline



#setting figure size

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20,10



#for normalizing data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))



#read the file

df = pd.read_csv('../input/stock_price_dada.csv')



#print the head

df.head(10)
#setting index as date

df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')

df.index = df['Date']
df.head()
#plot

plt.figure(figsize=(12,6))

plt.plot(df['Close'], label='Close Price history')
#creating dataframe with date and the target variable

data = df.sort_index(ascending=True, axis=0)

new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])



for i in range(0,len(data)):

     new_data['Date'][i] = data['Date'][i]

     new_data['Close'][i] = data['Close'][i]
data.head()
new_data.head()
#splitting into train and validation

train = new_data[:987]

valid = new_data[987:]
train.head()
valid.head()
new_data.shape, train.shape, valid.shape
train['Date'].min(), train['Date'].max(), valid['Date'].min(), valid['Date'].max()
#make predictions

preds = []

for i in range(0,248):

    a = train['Close'][len(train)-248+i:].sum() + sum(preds)

    b = a/248

    preds.append(b)
#calculate rmse

rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))

rms
valid['Close'][:10], preds[:10]
#plot

valid['Predictions'] = 0

valid['Predictions'] = preds

plt.plot(train['Close'])

plt.plot(valid[['Close', 'Predictions']])
#setting index as date values

df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')

df.index = df['Date']

df.head()
#sorting

data = df.sort_index(ascending=True, axis=0)

data.head()
#creating a separate dataset

new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):

    new_data['Date'][i] = data['Date'][i]

    new_data['Close'][i] = data['Close'][i]

    

new_data.head()