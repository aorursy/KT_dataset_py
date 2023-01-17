import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd

import seaborn as sns



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import adfuller, acf, pacf
with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")
instruments = sorted(df['id'].unique())
target_df = df[df['id'] == ids[0]][['timestamp', 'y']].set_index('timestamp')
def is_stationary(data):

    dftest = adfuller(data)

    

    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    # for k,v in dftest[4].items():

    #    dfoutput['Critical Value ({})'.format(k)] = v

    # print(dfoutput)

    

    if dftest[0] > dftest[4]['5%']:

        return False

    

    # print(dftest[0], 'is less than', dftest[4]['5%'])

    return True

nonstationary_ids = []

for id in ids:

    target_df = df[df['id'] == id][['timestamp', 'y']].set_index('timestamp')
x = is_stationary(target_df['y'].values)
target_df.plot()
plt.plot(pd.rolling_mean(target_df, window=10), label='rolling mean (10)')

plt.plot(pd.rolling_std(target_df, window=10), label='rolling std (10)')

plt.legend()
print(plot_acf(target_df, lags=50, alpha=.05))
print(plot_pacf(target_df, lags=50))
x = acf(target_df, alpha=.05)

x
dftest = adfuller(target_df['y'].values)

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for k,v in dftest[4].items():

    dfoutput['Critical Value ({})'.format(k)] = v

print(dfoutput)