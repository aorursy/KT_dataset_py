import pandas as pd

import numpy as np

import os 

import matplotlib.pyplot as plt

import seaborn as sn

data = pd.read_csv('../input/vgsales.csv')
data.head(10)
data.dtypes
data.shape
data.isnull().sum()
data.isnull().sum()
data.Genre.value_counts()
EU = data.pivot_table('EU_Sales', columns='Genre', aggfunc='sum').T

EU = EU.sort_values(by='EU_Sales', ascending=False).iloc[:3]

EU_genres = EU.index



JP = data.pivot_table('JP_Sales', columns='Genre', aggfunc='sum').T

JP = JP.sort_values(by='JP_Sales', ascending=False).iloc[:3]

JP_genres = JP.index



NA = data.pivot_table("NA_Sales", columns='Genre', aggfunc='sum').T

NA = NA.sort_values(by="NA_Sales", ascending=False).iloc[:3]

NA_genres = NA.index



Other = data.pivot_table('Other_Sales', columns='Genre', aggfunc='sum').T

Other = Other.sort_values(by='Other_Sales', ascending=False).iloc[0:3]

Other_genres = Other.index



colors =  {'Action':"orchid", 'Sports':"mediumslateblue", 'Shooter':"cornflowerblue", 'Role-Playing':"steelblue"}

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,4,1)

ax1.set_xticklabels(labels = EU_genres, rotation=90, size=14)

sn.barplot(x=EU_genres, y=EU["EU_Sales"], palette=colors)

plt.title('European Union', size=18)

plt.ylabel('Revenue in $ Millions', size=16)



ax2 = fig.add_subplot(1,4,2, sharey=ax1)

ax2.set_xticklabels(labels = JP_genres, rotation=90, size=14)



sn.barplot(x=JP_genres, y=JP['JP_Sales'], palette=colors)

plt.title('Japan', size=18)



ax3 = fig.add_subplot(1,4,3, sharey=ax1)

ax3.set_xticklabels(labels = NA_genres, rotation=90, size=14)



sn.barplot(x=NA_genres, y=NA['NA_Sales'], palette=colors)

plt.title('North America', size=18)



ax4 = fig.add_subplot(1,4,4, sharey=ax1)

ax4.set_xticklabels(labels = Other_genres, rotation=90, size=14)



sn.barplot(x=Other_genres, y=Other['Other_Sales'], palette=colors)

plt.title('Other', size=18)

plt.suptitle('Top 3 Genres by Revenue Per Region', size=22)

plt.show();
EU = data.pivot_table('EU_Sales', columns='Year', aggfunc='sum').T

EU = EU.sort_values(by='EU_Sales', ascending=False).iloc[0:3]

EU_years = EU.index.astype(int)



JP = data.pivot_table('JP_Sales', columns='Year', aggfunc='sum').T

JP = JP.sort_values(by='JP_Sales', ascending=False).iloc[0:3]

JP_years = JP.index.astype(int)



NA = data.pivot_table('NA_Sales', columns='Year', aggfunc='sum').T

NA = NA.sort_values(by='NA_Sales', ascending=False).iloc[0:3]

NA_years = NA.index.astype(int)



Other = data.pivot_table('Other_Sales', columns='Year', aggfunc='sum').T

Other = Other.sort_values(by='Other_Sales', ascending=False).iloc[0:3]

Other_years = Other.index.astype(int)



colors =  {2006.0:"yellowgreen", 2007.0:"mediumaquamarine", 2008.0:"cornflowerblue", 2009.0:"steelblue", 2010.0:"darkseagreen"}

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,4,1)

ax1.set_xticklabels(labels = EU_years, rotation=90, size=14)



sn.barplot(x=EU_years, y=EU['EU_Sales'], palette=colors)

plt.title('European Union', size=18)

plt.ylabel('Revenue in $ Millions', size=16)



ax2 = fig.add_subplot(1,4,2, sharey=ax1)

ax2.set_xticklabels(labels = JP_years, rotation=90, size=14)



sn.barplot(x=JP_years, y=JP['JP_Sales'], palette=colors)

plt.title('Japan', size=18)



ax3 = fig.add_subplot(1,4,3, sharey=ax1)

ax3.set_xticklabels(labels = NA_years, rotation=90, size=14)

sn.barplot(x=NA_years, y=NA['NA_Sales'], palette=colors)

plt.title('North America', size=18)



ax4 = fig.add_subplot(1,4,4, sharey=ax1)

ax4.set_xticklabels(labels = Other.index, rotation=90, size=14)

sn.barplot(x=Other_years, y=Other['Other_Sales'], palette=colors)

plt.title('Other', size=18)

plt.suptitle('Top 3 Years by Revenue Per Region', size=22)

plt.show();
y = data.groupby(['Year']).sum()

y = y['Global_Sales']

x = y.index.astype(int)



plt.figure(figsize=(12,8))

ax = sn.barplot(y = y, x = x)

ax.set_xlabel(xlabel='$ Millions', fontsize=16)

ax.set_xticklabels(labels = x, fontsize=12, rotation=50)

ax.set_ylabel(ylabel='Year', fontsize=16)

ax.set_title(label='Game Sales in $ Millions Per Year', fontsize=20)

plt.show();
df = data.Publisher.value_counts().sort_values(ascending=False).head(10)

df = pd.DataFrame(df)

publishers = df.index

df.columns = ['Releases']

df

plt.figure(figsize=(18,8))

ax = sn.barplot(x=publishers, y='Releases', data = df)

ax.set_xlabel(xlabel='Publishers')
df2 = data.Genre.value_counts().sort_values(ascending=False).head(10)

df2 = pd.DataFrame(df2)

genres = df2.index

df2.columns = ['Games']



plt.figure(figsize=(14,8))



colors = sn.color_palette("RdBu", n_colors=10)

ax = sn.barplot(x=genres,y='Games', data=df2, palette=colors)

ax.set_xlabel(xlabel='Genre')
sn.distplot(data['Year'],

                 kde=True,

                 rug = False,         # Show vertical lines at bottom, density of points wise

                 bins = 70

                 )
data.Genre.value_counts().sort_values().head()




# Sunset data by Genre

pdata = data[data['Genre'] == 'Puzzle']

sdata = data[data['Genre'] == 'Strategy']

fdata = data[data['Genre'] == 'Fighting']

sidata = data[data['Genre'] == 'Simulation']

pldata = data[data['Genre'] == 'Platform']



sn.jointplot( "Year", "Global_Sales",data=pldata, color='blue')



sn.jointplot("Year", "Global_Sales",data=fdata, color='red')



sn.jointplot("Year", "Global_Sales",data=sidata, color='yellow')



sn.jointplot("Year", "Global_Sales",data=sdata, color='pink')



sn.jointplot("Year", "Global_Sales",data=pdata, color='green')

# Year and Global_Sales relationship Genre-wise - in top 5 genres



fig, axs = plt.subplots(ncols=5,figsize=(20,6))



sn.regplot(x='Year', y='Global_Sales', data=pdata, ax=axs[0])

axs[0].set_title('Puzzle', fontsize=18)



sn.regplot(x='Year', y='Global_Sales', data=pldata, ax=axs[1])

axs[1].set_title('Platform', fontsize=18)



sn.regplot(x='Year',y='Global_Sales', data=sdata, ax=axs[2])

axs[2].set_title('Strategy', fontsize=18)



sn.regplot(x='Year',y='Global_Sales', data=sidata, ax=axs[3])

axs[3].set_title('Simulation', fontsize=18)



sn.regplot(x='Year',y='Global_Sales', data=fdata, ax=axs[4])

axs[4].set_title('Fighting', fontsize=18)

df5 = data.drop('Rank', axis=1)

sn.pairplot(df5,hue='Genre')