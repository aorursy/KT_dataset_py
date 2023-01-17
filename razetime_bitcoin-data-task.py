import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

import os

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex6 import *

print("Setup Complete")
bitcoin_filepath = "../input/bitcoin-data-from-2014-to-2020/BTC-USD.csv"

bitcoin_data = pd.read_csv(bitcoin_filepath, index_col='Date', parse_dates=True)

plt.figure(figsize=(24,16))

plt.title("Graph of Bitcoin Price over the Years 2014-2020",fontsize=30)

plt.ylabel("Price of Bitcoin (USD)")

bitcoin_data_dropped = bitcoin_data.drop(['Volume'], axis=1)

sns.lineplot(data=bitcoin_data_dropped)

sns.set_style("dark")
plt.figure(figsize=(24,16))

plt.title("Graph of Bitcoin Volume Over The Years 2014-2020", fontsize=30)



bitcoin_data_vol = bitcoin_data['Volume']

sns.lineplot(data=bitcoin_data_vol)

sns.set_style("dark")