import os

import json

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# Read the file

df = pd.read_csv('../input/maps-for-heroes-of-might-and-magic/hmm maps.csv')



# Make use of 'id' field and convert 'upload date' to proper datetime

df = df.set_index('id').sort_index()

df['upload_date'] = pd.to_datetime(df['upload_date'])



# Check

df.head()
# Check the downloads distribution

df['downloads'].hist(bins=100)



# See that there are outliers larget than approx. 100k downloads
# Plot downloads by author's country of origin

df[df['downloads']<100000].boxplot(column='downloads', by='country', rot=90, figsize=(20,5))
# Plot downloads by upload date

x = df['upload_date']

y = df['downloads']

plt.scatter(x, y, s=1)
# Try to spot main support line for amount download 

x = df[df['downloads']<2000]['upload_date']

y = df[df['downloads']<2000]['downloads']

plt.scatter(x, y, s=1)