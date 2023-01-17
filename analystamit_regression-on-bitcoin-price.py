# Load Dependancy
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
bit_df = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv')
bit_df.head()
bit_df['date'] = pd.to_datetime(bit_df.Timestamp, unit='s')
bit_df = bit_df.set_index('date')
# Rename columns so easier to code
bit_df = bit_df.rename(columns={'Open':'open', 'High': 'hi', 'Low': 'lo', 
                       'Close': 'close', 'Volume_(BTC)': 'vol_btc',
                       'Volume_(Currency)': 'vol_cur', 
                       'Weighted_Price': 'wp', 'Timestamp': 'ts'})
# Resampling
bit_df = bit_df.resample('d').agg({'open': 'mean', 'hi': 'mean', 
    'lo': 'mean', 'close': 'mean', 'vol_btc': 'sum',
    'vol_cur': 'sum', 'wp': 'mean', 'ts': 'min'})
# drop last row as it is not complete
bit_df = bit_df.iloc[:-1]
bit_df.describe()
bit_df['close'].plot(figsize=(14,10))
bit_df.plot(kind='scatter', x='ts', y='open', figsize=(14,10))
bit_df.dropna(axis = 0, how = 'any', inplace = True)
# Creating input (X) and labelled data (y) to train our model
X = bit_df[['ts']]
y = bit_df.close
from sklearn import linear_model
lr_model = linear_model.LinearRegression()
lr_model.fit(X, y)
pred = lr_model.predict(X)
ax = bit_df.plot(kind='scatter', x='ts', y='open', color='black', figsize=(14,10))
ax.plot(X, pred, color='blue')  
ax.plot(X, X*lr_model.coef_ + lr_model.intercept_+ 100, linestyle='--', color='green')
from sklearn.metrics import mean_squared_error
mean_squared_error(y, pred)
# R2 score
from sklearn.metrics import r2_score
print(r2_score(y, pred))

# plotting result
y_df = pd.DataFrame(y)
y_df['pred'] = pred
y_df['err'] = y_df.pred - y_df.close
(y_df
 .plot(figsize=(14,10))
)
X = bit_df.drop('close', axis = 1)
y = bit_df.close
cols = X.columns
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X)
X = ss.transform(X)
X = pd.DataFrame(X, columns=cols)
X.describe()
lr_model2 = linear_model.LinearRegression()
lr_model2.fit(X, y)
pred = lr_model2.predict(X)
lr_model2.score(X, y)
# plot result
y_df = pd.DataFrame(y)
y_df['pred'] = pred
y_df['err'] = y_df.pred - y_df.close
y_df.plot(figsize=(14,10))
# our scores get worse with recent data
lr_model2.score(X[-50:], y[-50:])
lr_model2.coef_
list(zip(X.columns, lr_model2.coef_))
# These coefficients correspond to the columns in X
pd.DataFrame(list(zip(X.columns, lr_model2.coef_)), columns=['Feature', 'Coeff'])
bit_df.plot(kind='scatter', x='wp', y='close', figsize=(14,10))
bit_df.plot(kind='scatter', x='vol_cur', y='close', figsize=(14,10))
