# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

sns.set(style="darkgrid")



import folium

import geopandas
df = pd.read_csv('/kaggle/input/covid19spain/covid19_Spain.csv', parse_dates = [7])
df.head()
df.isna().sum()
df.dtypes
plt.figure(figsize = (10,10))

sns.lineplot(x='Date', y='Total Infected',

             hue='CCAA',

             data=df)

plt.title('CCAA: Total Cases')
plt.figure(figsize = (10,10))

sns.lineplot(x='Date', y='UCI',

             hue='CCAA',

             data=df)

plt.title('CCAA: Total UCI')
plt.figure(figsize = (10,10))

sns.lineplot(x='Date', y='Hospitalized',

             hue='CCAA',

             data=df)

plt.title('CCAA: Total in Hospitals')
plt.figure(figsize = (10,10))

sns.lineplot(x='Date', y='Death',

             hue='CCAA',

             data=df)

plt.title('CCAA: Total Deaths')
plt.figure(figsize = (10,10))

sns.lineplot(x='Date', y='IA',

             hue='CCAA',

             data=df)

plt.title('CCAA: Total Deaths')
df_last = df[df['Date'] == df['Date'].max()]
df_last = df_last.sort_values(by='IA', ascending = False)
ccaa = df_last['CCAA'].head(5)
print('The top 5 affected CCAA according to IA are: ',ccaa.values)
df_last['Home'] = df_last['Total Infected']-df_last['Hospitalized']-df_last['UCI'] - df_last['Death']
# Pie chart

labels =list(['Home', 'Death', 'UCI','Hospitalized'])

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#fccf99']



for ca in ccaa:

    sizes = df_last[df_last['CCAA']==ca][labels].values[0]

    fig1, ax1 = plt.subplots()

    patches, texts, autotexts = ax1.pie(sizes, colors = colors,labels =labels, autopct='%1.1f%%', startangle=90)

    for text in texts:

        text.set_color('grey')

    for autotext in autotexts:

        autotext.set_color('grey')

    # Equal aspect ratio ensures that pie is drawn as a circle

    ax1.axis('equal')  

    plt.title(ca)

    plt.tight_layout()

    plt.show()