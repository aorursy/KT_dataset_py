"""

libraries and data import

"""

import numpy as np

import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



"""

enums

""" 

ID = "id"

TIMESTAMP = "timestamp"

Y = 'y'
# read data

with pd.HDFStore('../input/train.h5') as train:

    df = train.get('train')

df.set_index(["timestamp", "id"]);
# get only data for 200 longest living IDs, cut two slices ydfa and ydfb from somewhere within

# code shamelessly adapted from a notebook I do not remember (sorry!)

temp = df.groupby('id').apply(len)

temp = temp.sort_values()

temp = temp.reset_index()

temp2 = df[df['id'].isin(temp['id'].tail(200).values)]

temp2a = temp2[(temp2['timestamp'] <= 800 ) & ( temp2['timestamp'] >  600) ]

temp2b = temp2[(temp2['timestamp'] > 800  ) & ( temp2['timestamp'] < 1000) ]



ydfa = temp2a.pivot(index='timestamp', columns='id', values='y')

ydfb = temp2b.pivot(index='timestamp', columns='id', values='y')
# correlation matrix for each slice

corrmata = ydfa.corr()

corrmatb = ydfb.corr()



# plot 0

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmata, vmax=.8, square=True)

f.tight_layout()



# plot 1

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmatb, vmax=.8, square=True)

f.tight_layout()