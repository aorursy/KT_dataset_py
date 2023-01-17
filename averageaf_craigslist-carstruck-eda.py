# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df1= pd.read_csv('../input/craigslistVehiclesFull.csv')
df1.head()
initshape = df1.shape
print("The max and min respectively")

print(df1['price'].max())

print(df1['price'].min())

print(df1.describe()['price'])

print(df1.quantile(q=[0.01,0.05,0.8,0.85,0.90,0.95,0.99])['price'])
df1 = df1.drop(df1[df1.price > 75000].index)

secondshape = df1.shape

print("Listings removed >75000 :")

print(initshape[0] - df1.shape[0])

df1 = df1.drop(df1[df1.price < 250].index)

print("Listings removed == 0 :")

print(secondshape[0] - df1.shape[0])



plt.figure(figsize=(9,9))

sns.heatmap(df1.isnull(),cmap="Pastel1");
del df1['condition']

del df1['odometer']

del df1['cylinders']

del df1['vin']

del df1['drive']

del df1['size']

del df1['type']

del df1['paint_color']
plt.figure(figsize=(24,6))

ax = sns.countplot(x='manufacturer',data=df1,order=df1['manufacturer'].value_counts().index[:30])

locs, labels = plt.xticks();

plt.setp(labels, rotation=45);

ax.set_title("Listing Count per Manufacturer - top 30");





x = df1.price



f, (ax_hist, ax_box) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.75, .25)},figsize=(25,8))

sns.distplot(x,bins=75,ax=ax_hist)

sns.boxplot(x,ax= ax_box,fliersize = 0)

ax_hist.set(title="Price distribution of car listings",xlabel = '');

print('Unique years in dataset')

print(df1.year.unique())

df1.drop(df1[df1.year.isnull()].index, inplace = True)

df1.drop(df1[df1.year < 1960].index, inplace = True)
plt.figure(figsize=(24,10))

ax3 = sns.distplot(df1.year)

ax3.set(title="Distribution of Year in car listings");
print("Top 10 model years of listings")

print(df1.year.value_counts().iloc[:10])
statecount = df1.state_code.value_counts()
datamap = dict(type='choropleth',

            colorscale = 'Reds',

            locations = statecount.index,

            z = statecount,

            locationmode = 'USA-states',

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),

            colorbar = {'title':"Cars listed per State"}

            ) 
layout = dict(title = 'Cars listed per State',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )
choromap = go.Figure(data = [datamap],layout = layout)

iplot(choromap)
medpriceXX = df1.groupby('state_code')['price'].median()
datamap2 = dict(type='choropleth',

            colorscale = 'Portland',

            locations = medpriceXX.index,

            z = medpriceXX,

            locationmode = 'USA-states',

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),

            colorbar = {'title':"Median Car Price per State"}

            ) 

layout2 = dict(title = 'Median Car Price per State',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )
choromap2 = go.Figure(data = [datamap2],layout = layout2)

iplot(choromap2)