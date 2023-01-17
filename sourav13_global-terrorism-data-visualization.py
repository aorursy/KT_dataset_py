# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import matplotlib.pyplot as plt



#plotly

import plotly.plotly as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

import plotly.io as pio

init_notebook_mode(connected=True)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')
df.head()  #gives us top 5 rows of the dataframe
df.describe()
df.isnull().count()
df.isnull().sum() #gives us total no. of missing values for all the columns
totalms = df.isnull().sum().sort_values(ascending=False)

percentagems = (df.isnull().sum()/ df.isnull().count()).sort_values(ascending=False)

missingdata = pd.concat([totalms, percentagems], axis=1, keys=['Total', 'Percentage'])

missingdata.head()
print(df.shape)
print(missingdata.shape)
print(missingdata)
fig, ax = plt.subplots(figsize=(20,10))

ax = sns.countplot(x='iyear', palette='GnBu_d', data=df, orient="v")

_ = plt.xlabel('Years')

_ = plt.setp(ax.get_xticklabels(), rotation = 90)
df['country_txt'].value_counts()
fig, ax = plt.subplots(figsize=(15,10))

ax = sns.countplot(x='country_txt', data=df, order=df['country_txt'].value_counts()[:15].index, palette='inferno')

_ = plt.xlabel('Countries')

_ = plt.setp(ax.get_xticklabels(), rotation = 90)
df['region_txt'].value_counts()
df['region_txt'].isna().sum()
fig, ax = plt.subplots(figsize=(15,10))

ax = sns.countplot(x='region_txt', data=df, palette='plasma', order=df['region_txt'].value_counts().index)

_ = plt.xlabel('Region')

_ = plt.setp(ax.get_xticklabels(), rotation = 60)
terror_region = pd.crosstab(df['iyear'], df['region_txt'])

terror_region.plot(color=sns.color_palette('viridis', 12))

fig = plt.gcf()

fig.set_size_inches(10,6)

#use plotly
print(terror_region.head())
sns.countplot(x='success', data=df, palette='hls')
sns.countplot(x='suicide', data=df, palette='twilight')
df['attacktype1_txt'].isnull().sum()
fig, ax = plt.subplots(figsize=(15,10))

ax = sns.countplot(x='attacktype1_txt', data=df, palette='plasma_r', order=df['attacktype1_txt'].value_counts().index)

_ = plt.xlabel('AttackType')

_ = plt.setp(ax.get_xticklabels(), rotation = 75)
terror_attack = pd.crosstab(df['iyear'], df['attacktype1_txt'])

terror_attack.plot(color = sns.color_palette('Set3', 9))

fig = plt.gcf()

fig.set_size_inches(10, 6)
df['attacktype2_txt'].value_counts()
df['attacktype3_txt'].value_counts()
df['attacktype1_txt'].isna().sum()
percentage_missing = (df['attacktype2_txt'].isna().sum() / df['attacktype2_txt'].isna().count()) * 100.00

print(percentage_missing)
percentage_missing2 = (df['attacktype3_txt'].isna().sum() / df['attacktype3_txt'].isna().count()) * 100.00

print(percentage_missing2)
df['targtype1_txt'].value_counts()
fig, ax = plt.subplots(figsize=(15,10))

ax = sns.countplot(x = 'targtype1_txt', data=df, palette='icefire', order=df['targtype1_txt'].value_counts().index)

_ = plt.xlabel('Targets of attack')

_ = plt.setp(ax.get_xticklabels(), rotation = 90)
df['weaptype1_txt'].value_counts()
df['weaptype1_txt'].isna().sum()
fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.countplot(x='weaptype1_txt', data=df, palette='inferno', order=df['weaptype1_txt'].value_counts().index)

_ = plt.xlabel('Weapon Used')

_ = plt.setp(ax.get_xticklabels(), rotation = 90)
df['gname'].value_counts()[0:1]
fig, ax = plt.subplots(figsize=(15,10))

ax = sns.countplot(x='gname', data=df, palette='inferno', order=df['gname'].value_counts()[1:16].index)

_ = plt.xlabel('Terrorist group name')

_ = plt.setp(ax.get_xticklabels(), rotation=90) 
df['casualities'] = df['nkill'] + df['nwound']
df_country_cas = pd.concat([df['country_txt'], df['casualities']], axis=1)

df3 = pd.DataFrame(df_country_cas.groupby('country_txt').sum().reset_index())

df3.head()
print(df3.shape)
x = df3['country_txt']

y = df3['casualities']

sz = 10

colors = np.random.randn(205)

fig = go.Figure()

fig.add_scatter(

    x = x,

    y = y, 

    mode = 'markers',

    marker={

        'size':sz,

        'color':colors,

        'opacity':0.6,

        'colorscale':'Viridis'

    });

iplot(fig)
missingdata.loc['nkill', :]
missingdata.loc['nwound', :]
df_year_kill = pd.concat([df['iyear'], df['nkill']], axis=1)
df2 = pd.DataFrame(df_year_kill.groupby('iyear').sum().reset_index())

df2.head()
print(df2.shape)
df_year_wound = pd.concat([df['iyear'], df['nwound']], axis=1)

df3 = pd.DataFrame(df_year_wound.groupby('iyear').sum().reset_index())

df3.head()


x = df2['iyear']

y = df2['nkill']

colors = np.random.randn(47)

sz = 15

fig = go.Figure()

fig.add_scatter(

    x = x, 

    y = y, 

    mode = 'markers', 

    marker = {

        'size':sz,

        'color':colors,

        'opacity':0.6,

        'colorscale':'Viridis'

    });

iplot(fig)
x1 = df3['iyear']

y1 = df3['nwound']

colors = np.random.randn(47)

sz = 15

fig = go.Figure()

fig.add_scatter(

    x = x1, 

    y = y1, 

    mode = 'markers',

    marker = {

        'size':sz,

        'color':colors,

        'opacity':0.6,

        'colorscale':'Viridis'

    });

iplot(fig)