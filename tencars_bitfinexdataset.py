import os
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
# Data path
#input_path = "../input/cryptominuteresolution"
input_path = "../input"

# Get names and number of available currency pairs
pair_names = [x[:-4] for x in os.listdir(input_path)]
n_pairs = len(pair_names)

# Print the first 50 currency pair names
print("These are the first 50 out of {} currency pairs in the dataset:".format(n_pairs))
print(pair_names[0:50])
def load_data(symbol, source=input_path):
    path_name = source + "/" + symbol + ".csv"
    
    # Load data
    df = pd.read_csv(path_name, index_col='time')
    
    # Convert timestamp to datetime
    df.index = pd.to_datetime(df.index, unit='ms')
    
    # As mentioned in the description, bins without any change are not recorded.
    # We have to fill these gaps by filling them with the last value until a change occurs.
    df = df.resample('1T').pad()
    
    return df
btcusd = load_data("btcusd")
ethusd = load_data("ethusd")
ltcusd = load_data("ltcusd")
xrpusd = load_data("xrpusd")

# Take a look at the head of the BTC/USD data
btcusd.head()
# Set up the sub plots
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Plot the data
ax[0].plot(btcusd['close'], color='blue', label='BTC/USD')
ax[0].set_ylabel('price [USD]', fontsize=20, color='blue')
ax[0].set_xlabel('date', fontsize=20)
ax[0].set_title('entire data', fontsize=23)
ax[0].set_xlim([btcusd.index[0], btcusd.index[-1]])
ax[0].grid()

ax[0].twinx().plot(ethusd['close'], color='red', label='ETH/USD')

ax[1].plot(btcusd['close'].iloc[-1000:], color='blue', label='BTC/USD')
ax[1].set_ylabel('price [USD]', fontsize=20, color='blue')
ax[1].set_xlabel('date', fontsize=20)
ax[1].set_title('last 1000 data points', fontsize=23)
ax[1].set_xlim([btcusd.index[-1000], btcusd.index[-1]])
ax[1].grid()

ax[1].twinx().plot(ethusd['close'].iloc[-1000:], color='red', label='ETH/USD')

ax[2].plot(btcusd['close'].iloc[-100:], color='blue', label='BTC/USD')
ax[2].set_ylabel('price [USD]', fontsize=20, color='blue')
ax[2].set_xlabel('date', fontsize=20)
ax[2].set_title('last 100 data points', fontsize=23)
ax[2].set_xlim([btcusd.index[-100], btcusd.index[-1]])
ax[2].grid()

ax[2].twinx().plot(ethusd['close'].iloc[-100:], color='red', label='ETH/USD')

fig.tight_layout()
plt.show()
def log_return(data):
    return np.log(data.shift(0)/data.shift(1))
# Define the parameters for the moving mean
n_ticks = 1200
rolling_mean = 120

fig, ax = plt.subplots(2, 1, figsize=(18, 10))

ax[0].plot(log_return(xrpusd['close'].iloc[-n_ticks:]), alpha=0.5, label='xrpusd')
ax[0].plot(log_return(ltcusd['close'].iloc[-n_ticks:]), alpha=0.5, label='ltcusd')
ax[0].plot(log_return(btcusd['close'].iloc[-n_ticks:]), alpha=0.5, label='btcusd')
ax[0].plot(log_return(ethusd['close'].iloc[-n_ticks:]), alpha=0.5, label='ethusd')

ax[0].set_xlim([ethusd.index[-1200], ethusd.index[-1]])
ax[0].set_title('log returns', fontsize=23)
ax[0].set_xlabel('date', fontsize=20)
ax[0].grid()
ax[0].legend()

ax[1].plot(log_return(xrpusd['close'].iloc[-n_ticks:].rolling(rolling_mean).mean()), alpha=0.5, label='xrpusd')
ax[1].plot(log_return(ltcusd['close'].iloc[-n_ticks:].rolling(rolling_mean).mean()), alpha=0.5, label='ltcusd')
ax[1].plot(log_return(btcusd['close'].iloc[-n_ticks:].rolling(rolling_mean).mean()), alpha=0.5, label='btcusd')
ax[1].plot(log_return(ethusd['close'].iloc[-n_ticks:].rolling(rolling_mean).mean()), alpha=0.5, label='ethusd')

ax[1].set_xlim([ethusd.index[-1200], ethusd.index[-1]])
ax[1].set_title('log returns with {} min moving mean'.format(rolling_mean), fontsize=23)
ax[1].set_xlabel('date', fontsize=20)
ax[1].grid()
ax[1].legend()

fig.tight_layout()
plt.show()
# Load currency pairs and calculate the moving mean
data = pd.DataFrame()
col_names = []
for pair in pair_names[0:10]:
    tmp = load_data(pair)
    data = pd.concat([data, log_return(tmp.close).rolling(rolling_mean).mean()], axis=1)
    col_names.append(pair)
    
data.columns = col_names
data.dropna(how='all', axis=1, inplace=True)
corr_mat = data.corr()

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
sns.heatmap(corr_mat, vmin=-0.1, vmax=0.1, cmap="jet")
ax.set_title('Correlation map of currency pairs', fontsize=23)
plt.show()
