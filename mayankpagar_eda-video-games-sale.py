import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

from pandas_profiling import ProfileReport

import warnings

warnings.filterwarnings("ignore")

plt.style.use("fivethirtyeight")

sns.set_context("talk")

%matplotlib inline
df = pd.read_csv('../input/targetencodingsdata/vgsales1.csv')

df.head()
df.dropna(subset = ["Year"], inplace=True)

df = df[df.Year != 2017]

df = df[df.Year != 2020]

df.shape
df['Year'] = df['Year'].astype('int')

df_year = df['Year']

df_year = pd.DataFrame(df_year.value_counts())

df_year = df_year.sort_values('Year' , ascending=False).reset_index()

df_year.rename(columns = {'index':'Year','Year':'Frequency'}, inplace=True)

df_year['perc']= (df_year['Frequency']/df_year['Frequency'].sum())*100

top = df_year.head(10)

bottom = df_year.tail(10)

fig = plt.figure(figsize=(20,6))

ax=sns.barplot(x='Year', y='Frequency', data=top, palette='cool_r')

tt='Top 10 Most Frequent Years'

ax.set_title(tt, fontsize=25,y=1.1)

ax.set_ylabel('Frequency',fontsize=15)

ax.set_xlabel('Year',fontsize=15)

ax.axes.get_xaxis().set_visible(True)

l = top.sort_values('Year')['perc'].tolist()

j=0

for i in ax.patches:

    l[j] = round(l[j],1)

    ax.text(i.get_x() + 0.2, i.get_height() + 5, str(l[j]) + " %", fontsize = 14, color='black')

    j=j+1

df_sale=df[['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

df_sale.describe()
df_sale=df_sale.sum(axis=0)

df_sale= pd.DataFrame(df_sale)

df_sale.rename(columns = {0:'Total'}, index={'NA_Sales':'North America Sales',

                                             'EU_Sales':'Europe Sales',

                                             'JP_Sales':'Japan Sales ',

                                             'Other_Sales':'Other Sales'}, inplace=True)

df_sale.drop(['Global_Sales'], inplace = True) 

df_sale
colors_list = ['#7effdb','#b693fe', '#8c82fc', '#ff9de2']

mpl.rcParams['font.size'] = 18.0

df_sale['Total'].plot(kind='pie',

            figsize=(15, 6),

            autopct='%1.1f%%', 

            startangle=90,  

            shadow=True,       

            labels=None,    

            pctdistance=1.2, 

            colors=colors_list,

            )



plt.title('Global Sales',y=1.1,fontsize=25) 

plt.axis('equal') 

plt.legend(labels=df_sale.index, loc='upper left') 

plt.show()
df_game = pd.DataFrame(df['Name'].value_counts().head())

df_plat = pd.DataFrame(df['Platform'].value_counts().head())

ax=df_game.plot( kind='line', figsize=(6,5), marker='o', markerfacecolor='#848ccf', markersize=12, color='#be5683', linewidth=4)

plt.legend()

plt.xticks(rotation=45, ha='right')

plt.title('Most Frequent Games',y=1.1)

plt.ylabel('Count')

ax.axes.get_yaxis().set_visible(True)
ax=df_plat.plot( kind='line', figsize=(6,5), marker='o', markerfacecolor='#be5683', markersize=12, color='#848ccf', linewidth=4)

plt.legend()

plt.title('Most Frequent Platforms',y=1.1)

plt.ylabel('Count')

ax.axes.get_yaxis().set_visible(True)
df_gen = pd.DataFrame(df['Genre'].value_counts())

df_gen.plot(kind='bar',

            figsize=(15,7),

            width = 0.4,

            linewidth = 4,

            edgecolor = 'white',

            color = '#1f4068')

plt.xticks(rotation=45, ha='right')

plt.title('Genre Frequncy Distribution',fontsize=25,y=1.1)

plt.ylabel('Count')

ax.axes.get_yaxis().set_visible(True)
df_pub = pd.DataFrame(df['Publisher'].value_counts())

df_pub = df_pub.head(10)

df_pub.plot(kind='bar',

            figsize=(15,7),

            width = 0.4,

            linewidth = 4,

            edgecolor = 'white',

            color = '#fe8a71')

plt.xticks(rotation=45, ha='right')

plt.title('Publisher Frequency Distribution',y=1.1,fontsize=25)

plt.ylabel('Count')

ax.axes.get_yaxis().set_visible(True)
!jupyter nbextension enable --py --sys-prefix widgetsnbextension
import ipywidgets as widgets

from ipywidgets import HBox, VBox

from IPython.display import display
@widgets.interact_manual(

    Zone = ['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

def plot(Zone = 'Global_Sales',grid=True):

    df_top = df.sort_values(Zone)

    df_top = df_top[['Name',Zone]].set_index('Name')

    df_top = df_top.tail(10)

    ax = df_top.plot(kind='barh', 

          figsize = (10, 6), 

          width = 0.45,

          linewidth=3, 

          edgecolor='white',

          color='#848ccf')

    label = [ Zone + ' (in millions)']

    tt = "Top 10 Games with Highest Sales - " + Zone[:-6]

    ax.set_title(tt, fontsize=25,y=1.1)

    plt.legend(label,fontsize = 16)

    ax.set_yticklabels(df_top.index.tolist(),fontsize=22)

    ax.axes.get_yaxis().set_visible(True)
df_platsale = df[['Platform','NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

df_platsale = df_platsale.groupby('Platform')['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'].sum()

@widgets.interact_manual(

    Zone = ['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

def plot(Zone = 'Global_Sales'):

    ax = df_platsale[Zone].plot(color = '#6a197d',

                                      linewidth = 3,

                                      linestyle='--')

    ax = df_platsale[Zone].plot(kind='bar',

                                 figsize = (20,8),

                                 width = 0.2,

                                 color = '#ffa5b0',

                                 ax=ax)

    plt.xticks(rotation=45, ha='right',fontsize=20)

    tt = 'Number Of Sales Per Platform (in million) - ' + Zone[:-6]

    plt.title(tt,fontsize=30,y=1.1)

    ax.axes.get_yaxis().set_visible(True)
df_gsale = df[['Genre','NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

df_gsale = df_gsale.groupby('Genre')['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'].sum()

@widgets.interact_manual(

    Zone = ['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

def plot(Zone = 'Global_Sales'):

    ax = df_gsale[Zone].plot(kind='bar',

                                 figsize = (13,7),

                                 width = 0.4,

                                 linewidth = 4,

                                 edgecolor='white',

                                 color = '#32e0c4')

    plt.xticks(rotation=45, ha='right',fontsize=20)

    tt = 'Number Of Sales Per Genre (in million) - ' + Zone[:-6]

    plt.title(tt,fontsize=22,y=1.1)

    ax.axes.get_yaxis().set_visible(True)
@widgets.interact_manual(

    Zone = ['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'],

    Platform = list(df.Platform.unique()),

    Genre = list(df.Genre.unique()))

def plot(Zone = 'Global_Sales',Platform = 'X360', Genre='Action'):

    df_game = df[['Name','Platform','Genre','NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

    df_game = df_game[df_game['Platform']==Platform]

    df_game = df_game[df_game['Genre']==Genre]

    df_game = df_game.groupby('Name')['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'].sum()

    df_game = df_game.sort_values(Zone)

    df_game = df_game[Zone].tail()

    ax = df_game.plot(kind='barh', 

          figsize = (12, 4), 

          width = 0.45,

          linewidth=3, 

          edgecolor='white',

          color='#be9fe1')

    label = [ Zone + ' (in millions)']

    tt = "Top 5 Games With Highest Sales ( "+Platform+" ) ( "+Genre+" ) - "  + Zone[:-6]

    ax.set_title(tt, fontsize=25,y=1.1)

    plt.legend(label,fontsize = 16)

    ax.set_yticklabels(df_game.index.tolist(),fontsize=22)

    ax.axes.get_yaxis().set_visible(True)
@widgets.interact_manual(

    Zone = ['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

def plot(Zone = 'Global_Sales'):

    df_pub = df[['Publisher','NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

    df_pub = df_pub.groupby('Publisher')['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'].sum()

    df_pub = df_pub.sort_values(Zone)

    df_pub = df_pub.tail(15)

    ax = df_pub[Zone].plot(kind='bar',

                                 figsize = (15,7),

                                 width = 0.4,

                                 linewidth = 4,

                                 edgecolor='white',

                                 color = '#ff7272')

    plt.xticks(rotation=45, ha='right',fontsize=20)

    tt = 'Number Of Sales Per Publisher (in million) - ' + Zone[:-6]

    plt.title(tt,fontsize=30,y=1.1)

    ax.axes.get_yaxis().set_visible(True)
df_global = pd.DataFrame(df.groupby('Year')['Global_Sales'].sum())

ax=df_global.plot(kind='bar',

                    figsize = (20,8),

                    width = 0.4,

                    color = '#916dd5',

                    linewidth=2,

                    edgecolor= 'white')

plt.xticks(rotation=45, ha='right',fontsize=20)

ax.get_legend().remove()

tt = 'Number Of Sales By Year (in million)'

plt.title(tt,fontsize=30,y=1.1)

ax.axes.get_yaxis().set_visible(True)