# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

import plotly

import plotly.graph_objs as go

import os

from mpl_toolkits.basemap import Basemap # library to plot maps

data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding="ISO-8859-1")
data.info(verbose=True)

subData = pd.DataFrame(data[['iyear',

'crit2',

'crit3',

'doubtterr',

'success',

'suicide',

'attacktype1',

'guncertain1',

'nkill',

'nkillus',

'nkillter',

'nwound',

]]).corr()



sns.heatmap(subData,linewidths=.5)

data.plot(kind = 'scatter',x = 'iyear', y = 'nkill', alpha = 0.5 , color = 'red')
data.propextent.plot(kind = 'hist',bins = 25,figsize = [6,6])

gData = data[['country_txt','nkill']].groupby('country_txt').count().sort_values(by = ['nkill'], ascending=False)

fData = gData[(gData.nkill >1000)]

fData.nkill.plot(kind = 'bar' ,figsize = [15,5])
for x,y in data[data.iyear > 2016].head(1).iterrows():

        print ('Key: ',x , 'Value:',y)