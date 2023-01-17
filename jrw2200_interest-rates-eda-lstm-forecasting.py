import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns



import scipy.stats as stats

from scipy.stats import norm

from scipy.special import boxcox1p



import statsmodels

import statsmodels.api as sm

#print(statsmodels.__version__)



plt.style.use('fivethirtyeight') 
# import japan data

dfj = pd.read_csv('../input/japan-interest-rates/MOFJ-INTEREST_RATE_JAPAN.csv', parse_dates=['Date'], index_col = 'Date')

dfj.tail()
dfus = pd.read_csv('../input/fedselectedinterestrates/FRB_H15.csv')

dfus = dfus.iloc[5:]

dfus = dfus.rename(columns = {'Series Description':'Date',

        'Market yield on U.S. Treasury securities at 1-month   constant maturity, quoted on investment basis':'1M',

       'Market yield on U.S. Treasury securities at 3-month   constant maturity, quoted on investment basis':'3M',

       'Market yield on U.S. Treasury securities at 6-month   constant maturity, quoted on investment basis':'6M',

       'Market yield on U.S. Treasury securities at 1-year   constant maturity, quoted on investment basis':'1Y',

       'Market yield on U.S. Treasury securities at 2-year   constant maturity, quoted on investment basis':'2Y',

       'Market yield on U.S. Treasury securities at 3-year   constant maturity, quoted on investment basis':'3Y',

       'Market yield on U.S. Treasury securities at 5-year   constant maturity, quoted on investment basis':'5Y',

       'Market yield on U.S. Treasury securities at 7-year   constant maturity, quoted on investment basis':'7Y',

       'Market yield on U.S. Treasury securities at 10-year   constant maturity, quoted on investment basis':'10Y',

       'Market yield on U.S. Treasury securities at 20-year   constant maturity, quoted on investment basis':'20Y',

       'Market yield on U.S. Treasury securities at 30-year   constant maturity, quoted on investment basis':'30Y'})

dfus['Date']=pd.to_datetime(dfus['Date'])

dfus = dfus.set_index('Date')

for c in dfus.columns:

    dfus[c] = pd.to_numeric(dfus[c], errors='coerce')

dfus.tail()
for c in dfus.columns:

    dfus = dfus.rename(columns = {c: 'US_'+c})

dfus.tail()

for c in dfj.columns:

    dfj = dfj.rename(columns = {c: 'JPN_'+c})

dfj.head()



df = dfj.join(dfus, how='outer')



for c in dfus.columns:

    dfus = dfus.rename(columns = {c: c[3:]})



for c in dfj.columns:

    dfj = dfj.rename(columns = {c: c[4:]})

    

df.tail()
us_summ = dfus.describe()

us_summ = us_summ.transpose()

us_summ
j_summ = dfj.describe()

j_summ = j_summ.transpose()

j_summ
plt.plot(us_summ['max'])
fig, ax = plt.subplots(1, 2, figsize=(20,6))



ax[0].plot(us_summ['max'], color='red', linestyle = '-.', linewidth=1, label = 'max')

ax[0].plot( us_summ['mean'] + us_summ['std'], color='blue', linestyle = '-.', linewidth=1, label = 'mean + std')

ax[0].plot( us_summ['mean'], color='blue')

ax[0].plot(us_summ['mean'] - us_summ['std'], color='blue', linestyle = '-.', linewidth=1, label = 'mean - std')

ax[0].plot( us_summ['min'], color='red', linestyle = '-.', linewidth=1, label = 'min')

ax[0].fill_between(us_summ.index, us_summ['min'], us_summ['mean'] - us_summ['std'], color='r', alpha=.1)

ax[0].fill_between(us_summ.index, us_summ['mean'] + us_summ['std'], us_summ['max'], color='r', alpha=.1)

ax[0].fill_between(us_summ.index, us_summ['mean'] - us_summ['std'], us_summ['mean'] + us_summ['std'], color='b', alpha=.1)

ax[0].legend(fontsize=12, loc = 'upper left');

ax[0].set_ylim(-2.5, 18)

ax[0].set_xlabel('Bond duration')

ax[0].set_ylabel('Yield [%]')

ax[0].set_title('United States')



ax[1].plot(j_summ['max'], color='red', linestyle = '-.', linewidth=1, label = 'max')

ax[1].plot( j_summ['mean'] - j_summ['std'], color='blue', linestyle = '-.', linewidth=1, label = 'mean - std')

ax[1].plot( j_summ['mean'], color='blue')

ax[1].plot(j_summ['mean'] + j_summ['std'], color='blue', linestyle = '-.', linewidth=1, label = 'mean + std')

ax[1].plot( j_summ['min'], color='red', linestyle = '-.', linewidth=1, label = 'min')

ax[1].fill_between(j_summ.index, j_summ['min'], j_summ['mean'] - j_summ['std'], color='r', alpha=.1)

ax[1].fill_between(j_summ.index, j_summ['mean'] + j_summ['std'], j_summ['max'], color='r', alpha=.1)

ax[1].fill_between(j_summ.index, j_summ['mean'] - j_summ['std'], j_summ['mean'] + j_summ['std'], color='b', alpha=.1)

ax[1].legend(fontsize=12);

ax[1].set_ylim(-2.5, 18)

ax[1].set_xlabel('Bond duration')

ax[1].set_ylabel('Yield [%]')

ax[1].set_title('Japan')
fig, ax = plt.subplots(4,3,figsize=(24, 12))

for i, c in enumerate(dfus.columns):

    ax[i//3][i%3].plot(dfus[c], label = c )

    ax[i//3][i%3].legend()

    ax[i//3][i%3].set_ylabel('Yield [%]')

fig.delaxes(ax[3][2])
fig, ax = plt.subplots(5,3,figsize=(24, 12))

for i, c in enumerate(dfj.columns):

    ax[i//3][i%3].plot(dfj[c], label = c )

    ax[i//3][i%3].set_ylabel('Yield [%]')

    ax[i//3][i%3].legend()
f, ax = plt.subplots(1,2, figsize=(18,6))

corrmatrixus = dfus.corr()

sns.heatmap(corrmatrixus, square=True, ax=ax[0])

ax[0].set_title('US')

corrmatrixj = dfj.corr()

sns.heatmap(corrmatrixj, square=True, ax=ax[1])

ax[1].set_title('Japan')
dfus.columns
fig, ax = plt.subplots(2,3, figsize=(24, 12))

for i, c in enumerate(['30Y', '20Y', '10Y', '7Y', '5Y', '3Y']):

    ax[i//3][i%3].plot(dfus['1M'], dfus[c],'r+')

    ax[i//3][i%3].set_xlabel('Y$_{1M}$')

    ax[i//3][i%3].set_ylabel('Y$_{'+c+'}$')
fig, ax = plt.subplots(figsize=(18,5))

ax.plot(dfus['1M'])

ax.plot(dfus['30Y'])

ax.set_xlim('2000-01-01', '2020-01-01')

ax.set_ylim(0, 8)
dfj.columns
fig, ax = plt.subplots(2,3, figsize=(24, 12))

for i, c in enumerate(['1Y', '2Y', '3Y', '4Y', '5Y', '6Y']):

    ax[i//3][i%3].plot(dfj['30Y'], dfj[c],'k+')

    ax[i//3][i%3].set_xlabel('Y$_{30Y}$')

    ax[i//3][i%3].set_ylabel('Y$_{'+c+'}$')
fig, ax = plt.subplots(figsize=(18,5))

ax.plot(dfj['2Y'])

ax.plot(dfj['30Y'])

ax.set_xlim('1998-01-01', '2020-01-01')

ax.set_ylim(-0.5, 3)
fig, ax = plt.subplots(figsize=(18,5))

ax.plot(dfj['6Y'])

ax.plot(dfj['30Y'])

ax.set_xlim('1998-01-01', '2020-01-01')

ax.set_ylim(-0.5, 3)
f, ax = plt.subplots(figsize=(22,12))

corrmatrix = df.corr()

sns.heatmap(corrmatrix, square=True, ax=ax)

ax.set_title('US and Japan')
fig, ax = plt.subplots(figsize=(18,10.5))

diff = dfus['10Y'] - dfus['2Y']

plt.plot(dfus.index, diff)

plt.fill_between(dfus.index, 0, diff, where= diff < 0, color='red')

plt.fill_between(dfus.index, 0, diff, where= diff > 0, color='green')

plt.ylabel('Yield$_{10Y}$ - Yield$_{2Y}$')
fig, ax = plt.subplots(figsize=(18,10.5))

diff = dfus['3M'] - dfus['1M']

plt.plot(dfus.index, diff)

plt.fill_between(dfus.index, 0, diff, where= diff < 0, color='red')

plt.fill_between(dfus.index, 0, diff, where= diff > 0, color='green')

plt.ylabel('Yield$_{10Y}$ - Yield$_{2Y}$')
fig, ax = plt.subplots(figsize=(18,10.5))

diff = dfus['2Y'] - dfus['1M']

plt.plot(dfus.index, diff)

plt.fill_between(dfus.index, 0, diff, where= diff < 0, color='red')

plt.fill_between(dfus.index, 0, diff, where= diff > 0, color='green')

plt.ylabel('Yield$_{10Y}$ - Yield$_{2Y}$')
fig, ax = plt.subplots(figsize=(18,10.5))

diff = dfus['30Y'] - dfus['1Y']

plt.plot(dfus.index, diff)

plt.fill_between(dfus.index, 0, diff, where= diff < 0, color='red')

plt.fill_between(dfus.index, 0, diff, where= diff > 0, color='green')

plt.ylabel('Yield$_{30Y}$ - Yield$_{2Y}$')
## LSTM
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout

from keras import optimizers

from sklearn.preprocessing import MinMaxScaler
ts1 = dfus['5Y'].dropna()

ts1_idx = ts1.index



#fig, ax = plt.subplots(1,1,figsize=(24, 12))

#ax.plot(ts1, label = c )

#ax.set_ylabel('Yield [%]')

#ax.legend()
ts2 = ts1.values

scl = MinMaxScaler()

ts2 = ts2.reshape(ts2.shape[0],1)

ts2 = scl.fit_transform(ts2)
plt.plot(ts2)
def processData(data, lb):

    X, Y = [], []

    for i in range(len(data)-lb-1):

        X.append(data[i:(i+lb)])

        Y.append(data[i+lb])

    return np.array(X), np.array(Y)

X, y = processData(ts2, 6)

X_train, X_test = X[:int(X.shape[0]*0.80)], X[int(X.shape[0]*0.80):]

y_train, y_test = y[:int(y.shape[0]*0.80)], y[int(y.shape[0]*0.80):]

print('Checking that the split was performed correctly')

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
model = Sequential()

model.add(LSTM(32, input_shape=(6,1)))

model.add(Dropout(0.25))

model.add(Dense(1))

adam = optimizers.adam(lr=0.001, clipnorm=1.)

model.compile(optimizer=adam, loss='mse')

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

history = model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test), shuffle=False)
plt.plot(history.history['loss'], label = 'Training')

plt.plot(history.history['val_loss'], label = 'Validation')

plt.legend()
model64 = Sequential()

model64.add(LSTM(64, input_shape=(6,1)))

model64.add(Dropout(0.5))

model64.add(Dense(1))

adam = optimizers.adam(lr=0.001, clipnorm=1.)

model64.compile(optimizer=adam, loss='mse')

history64 = model64.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test), shuffle=False)
fig, ax = plt.subplots(1,1,figsize=(24, 12))

ax.plot(history64.history['loss'], label = 'Training')

ax.plot(history64.history['val_loss'], label = 'Validation')

ax.legend()
Xt32 = model.predict(X_test)

Xt64 = model64.predict(X_test)



fig, ax = plt.subplots(1,1,figsize=(24, 12))

ax.plot( scl.inverse_transform(y_test.reshape(-1,1)), label='Test data')

ax.plot( scl.inverse_transform(Xt32), label ='32L')

ax.plot( scl.inverse_transform(Xt64), label='64L')

ax.set_ylabel('Yield [%]')

ax.legend()