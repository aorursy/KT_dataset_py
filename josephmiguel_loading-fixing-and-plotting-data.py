%matplotlib inline

import pandas as pd

import numpy as np

import math

import seaborn as sns

import matplotlib.pyplot as plt

import time

from decimal import Decimal

import json

from decimal import *

import boto3

import datetime

import os

import string
!ls ../input/CryptoCurrencyHistoricalData
path = "../input/CryptoCurrencyHistoricalData/"

files = os.listdir(path=path)

data = pd.DataFrame()

for file in files:

    if not file.endswith('.csv'): continue

    # print(file)

    f = pd.read_csv(path + file, delimiter=';')

    f['coin'] = file.replace('.csv', '')

    data = data.append(f)

data.head()
# Identify bad data

bad_data = data[data.Date.map(lambda x: x.find('v')>0)].head()

bad_data
# fix bad data

bad_data['Date'] = bad_data['Date'].map(lambda x: x.replace('v', '/'))
fixed_data = data.copy()

fixed_data['Date'] = fixed_data['Date'].map(lambda x: x.replace('v', '/'))

fixed_data['Date'] = [datetime.datetime.strptime(_, '%d/%m/%Y') for _ in fixed_data['Date']]

fixed_data = fixed_data.sort_values(by='Date')

fixed_data['seconds'] = [int(_.timestamp()) for _ in fixed_data['Date']]

fixed_data.head()

# Plot out the results

plt.figure(figsize=(13, 13))

fixed_data[fixed_data.coin == 'bitcoin'].plot(y='Close')

plt.show()

sns.set(rc={"figure.figsize": (16, 16)})

sns.lmplot(data=fixed_data, x='seconds', y='Open', hue='coin', size=16)
