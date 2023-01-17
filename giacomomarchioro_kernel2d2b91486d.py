import pandas as pd

import matplotlib.pyplot as plt 

import numpy as np 

import seaborn as sns

import os
g = pd.read_csv(r"../input/isotc213/ISO_TC 213.csv")
g
g['Status'].value_counts().plot(kind='barh')

plt.ylabel("Number of standards")
stat = g['Status'].value_counts()

stat
op = g.price_CF.sum()

print( "Overall cost of published standard %s CF %s Euro" %(op,op*0.91))
g.Number_of_pages.sum()
g['year'] = (np.where(g['Publication_date'].str.contains('-'),

                  g['Publication_date'].str.split('-').str[0],

                  g['Publication_date']))
gs = g.sort_values(by=['year'])

years = gs['year'].value_counts(sort=False)
years[sorted(years.index)].plot(kind='bar')
h = pd.read_csv("../input/isotc213/ISO_TC 213_CH.csv")

h.describe()
op = h.price_CF.sum()

print( "Overall cost of published standard %s CF %s Euro" %(op,op*0.91))
h['Status'].value_counts().plot(kind='barh')

plt.ylabel("Number of standards")
h
stat = h['Status'].value_counts()

stat