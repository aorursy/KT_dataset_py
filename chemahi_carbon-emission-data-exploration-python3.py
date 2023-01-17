%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv('../input/MER_T12_06.csv')
data.head(15)
data.Value = pd.to_numeric(data.Value, errors='coerce')

data.YYYYMM = pd.to_datetime(data.YYYYMM, format='%Y%m', errors='coerce')

data = data.dropna()
fuels = data.groupby('Description')

fig, ax = plt.subplots(figsize=(20,9))

for desc, group in fuels:

    group.plot(x='YYYYMM', y='Value', label=desc, ax=ax, title='Carbon Emissions per Fuel')

    group.plot(x='YYYYMM', y='Value', title=desc)

#fuels.plot(x='YYYYMM', y='Value')
#Total emissions per fuel

values = data.groupby('Description')['Value'].sum()

values
values[:-1].plot.bar(title='Total Carbon Emissions per Fuel')
agg = data.iloc[:, 1:]

agg = agg.set_index('YYYYMM')

agg = agg.groupby(['Description', pd.TimeGrouper('A')])['Value'].sum()

fig2, ax2 = plt.subplots(figsize=(20,9))

agg.unstack(level=0).plot(kind='bar', ax=ax2)

    
#per fuel per year

agg.unstack(level=0).shape

for col in agg.unstack(level=0):

    fig, ax = plt.subplots(figsize=(20,9))

    agg[col].plot(kind='bar', title=col)