import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Set seed for reproducability 
np.random.seed(42)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
df = pd.concat([train,test],axis=0)
df.head(20)
df.describe()
df.fillna(0, inplace=True)
df.isnull().sum()
df.dtypes
df.head(10)
def scatterplots(feature, label):
    x = feature
    y = df['Weekly_Sales']
    plt.scatter(x, y)
    plt.ylabel('sales')
    plt.xlabel(label)
    plt.show()
    
fig = plt.gcf()
fig.set_size_inches(8, 8)
headers = list(df)
labels = headers
scatterplots(df['CPI'], 'CPI')
scatterplots(df['Date'], 'Date') # date isn't readable in scatterplot. make it timeseries
scatterplots(df['Dept'], 'Dept')
scatterplots(df['Fuel_Price'], 'Fuel_Price')
scatterplots(df['IsHoliday'], 'IsHoliday')
scatterplots(df['MarkDown1'], 'MarkDown1')
scatterplots(df['MarkDown2'], 'MarkDown2')
scatterplots(df['MarkDown3'], 'MarkDown3')
scatterplots(df['MarkDown4'], 'MarkDown4')
scatterplots(df['MarkDown5'], 'MarkDown5')
scatterplots(df['Size'], 'Size')
scatterplots(df['Store'], 'Store')
scatterplots(df['Temperature'], 'Temperature')
scatterplots(df['Type'], 'Type')
scatterplots(df['Unemployment'], 'Unemployment')
# extract month from date
df['month'] = pd.DatetimeIndex(df['Date']).month
df.head()
# create train2. take median and groupby. get mean_vals
train2 = train
train2['month'] = pd.DatetimeIndex(train2['Date']).month
mean_vals = train2.groupby(['Store', 'Dept', 'month', 'IsHoliday']).median()
# replace NaN
mean_vals.fillna(0, inplace=True)
mean_vals.dtypes
y1 = train2['Weekly_Sales'].values
X1 = train2.drop('Weekly_Sales',axis=1).values
model = Sequential()

# std 3 layer network
model.add(Dense(units=320, input_dim=16, activation='tanh'))
model.add(Dense(units=160, activation='tanh'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam',
              loss='mae',
              metrics=['acc'])

model.fit(X1, y1, epochs=10, batch_size= 2048)
