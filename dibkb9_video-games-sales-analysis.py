# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df_genre = df.groupby('Genre')
def genre_sales(region):

    xrange = np.arange(1,len(df_genre.sum())+1)

    fig,ax= plt.subplots(ncols=2,figsize=(18,6))

    df_to_plot = df_genre.sum().sort_values(by=region,ascending =False)[::-1]

    df_to_plot[region].plot(kind='barh')

    #labels

    ax[1].set_ylabel(None)

    ax[1].tick_params(axis='both', which='major', labelsize=13)

    ax[1].set_xlabel('Total Sales(in millions)', fontsize=15,labelpad=21)

    #spines

    ax[1].spines['top'].set_visible(False)

    ax[1].spines['right'].set_visible(False)

    ax[1].grid(False)

    #annotations    

    for x,y in zip(np.arange(len(df_genre.sum())+1),df_genre.sum().sort_values(by=region,ascending =False)[::-1][region]):

        label = "{:}".format(y)

        labelr = round(y,2)

        plt.annotate(labelr, # this is the text

                     (y,x), # this is the point to label

                      textcoords="offset points",# how to position the text

                     xytext=(6,0), # distance from text to points (x,y)

                    ha='left',va="center")

     

    #donut chart

    theme = plt.get_cmap('Blues')

    ax[0].set_prop_cycle("color", [theme(1. * i / len(df_to_plot))for i in range(len(df_to_plot))])    

    wedges, texts,_ = ax[0].pie(df_to_plot[region], wedgeprops=dict(width=0.45), startangle=-45,labels=df_to_plot.index,

                      autopct="%.1f%%",textprops={'fontsize': 13,})



 

    plt.tight_layout()    
genre_sales('Global_Sales')
genre_sales('NA_Sales')
genre_sales('EU_Sales')
genre_sales('JP_Sales')
df_platform = df.groupby('Platform')
def platform_sales(region):

    df_platform_plot = df_platform.sum().sort_values(by=region,ascending = False).head(12)[::-1]

    xrange = np.arange(1,len(df_platform_plot)+1)

    fig,ax = plt.subplots(ncols=2,figsize=(18,6))

    df_platform_plot[region].plot(kind='barh',color='#961515',alpha=.9)

    #labels

    ax[1].set_ylabel(None)

    ax[1].tick_params(axis='both', which='major', labelsize=13)

    ax[1].set_xlabel('Total Sales(in millions)', fontsize=15,labelpad=21)

    #spines

    ax[1].spines['top'].set_visible(False)

    ax[1].spines['right'].set_visible(False)

    ax[1].grid(False)

    #annotations    

    for x,y in zip( np.arange(len(df_platform_plot)+1),df_platform_plot[region]):

        label = "{:}".format(y)

        labelr = round(y,2)

        plt.annotate(labelr, # this is the text

                     (y,x), # this is the point to label

                      textcoords="offset points",# how to position the text

                     xytext=(6,0), # distance from text to points (x,y)

                     ha='left',va="center")

    #donut chart    

    theme = plt.get_cmap('Reds')

    ax[0].set_prop_cycle("color", [theme(1. * i / len(df_platform_plot))for i in range(len(df_platform_plot))])    

    wedges, texts,_ = ax[0].pie(df_platform_plot[region], wedgeprops=dict(width=0.45), startangle=-45,labels=df_platform_plot.index,

                      autopct="%.1f%%",textprops={'fontsize': 13,})

    

    plt.tight_layout()  
platform_sales('Global_Sales')
platform_sales('NA_Sales')
platform_sales('EU_Sales')
platform_sales('JP_Sales')
df_game_title = df.groupby('Name').sum()
df_game_title[['Global_Sales','NA_Sales','EU_Sales','JP_Sales']].sort_values(by='Global_Sales',

                                         ascending =False).head(12)[::-1].plot(kind='barh',figsize=(18,9),grid=False)
df_title = df.groupby('Name')
def title_sales(region):

    df_title_plot = df_title.sum().sort_values(by=region,ascending = False).head(12)[::-1]

    xrange = np.arange(1,len(df_title_plot)+1)

    fig,ax = plt.subplots(figsize=(15,6))

    df_title_plot[region].plot(kind='barh',color='#0aa391',alpha=.9)

    #labels

    ax.set_ylabel(None)



    ax.tick_params(axis='both', which='major', labelsize=13)

    ax.set_xlabel('Total Sales(in millions)', fontsize=15,labelpad=21)

    #spines

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.grid(False)

    #annotations    

    for x,y in zip( np.arange(len(df_title_plot)+1),df_title_plot[region]):

        label = "{:}".format(y)

        labelr = round(y,2)

        plt.annotate(labelr, # this is the text

                     (y,x), # this is the point to label

                      textcoords="offset points",# how to position the text

                     xytext=(6,0), # distance from text to points (x,y)

                     ha='left',va="center") 

    plt.tight_layout()    
title_sales('Global_Sales')
title_sales('NA_Sales')
title_sales('EU_Sales')
title_sales('JP_Sales')
df_publisher = df.groupby('Publisher')
def studio_sales(region):

    df_pub_plot = df_publisher.sum().sort_values(by=region,ascending = False).head(12)[::-1]

    xrange = np.arange(1,len(df_pub_plot)+1)

    fig,ax = plt.subplots(figsize=(15,6))

    df_pub_plot[region].plot(kind='barh',color='#469109',alpha=.9)

    #labels

    ax.set_ylabel(None)

    ax.tick_params(axis='both', which='major', labelsize=13)

    ax.set_xlabel('Total Sales(in millions)', fontsize=15,labelpad=21)

    #spines

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.grid(False)

    #annotations    

    for x,y in zip( np.arange(len(df_pub_plot)+1),df_pub_plot[region]):

        label = "{:}".format(y)

        labelr = round(y,2)

        plt.annotate(labelr, # this is the text

                     (y,x), # this is the point to label

                    textcoords="offset points",# how to position the text

                     xytext=(6,0), # distance from text to points (x,y)

                     ha='left',va="center")  
studio_sales('Global_Sales')
studio_sales('NA_Sales')
studio_sales('EU_Sales')
studio_sales('JP_Sales')
df_year = df.groupby('Year').sum().sort_values(by=['Year'],ascending = False)
#Dropped 2020 year because data of 2018,2019 were not present.

df_year.drop([2020.0],inplace =True)
fig,ax = plt.subplots(figsize=(18,6))

ax.plot(df_year.index,df_year['Global_Sales'],label ='Global',linewidth=3)

ax.plot(df_year.index,df_year['NA_Sales'],label ='North-America',linewidth=3)

ax.plot(df_year.index,df_year['EU_Sales'],label ='Europe',linewidth=3)

ax.plot(df_year.index,df_year['JP_Sales'],label ='Japan',linewidth=3)

ax.legend(loc="upper left")

ax.set_ylabel('Total Sales(in millions)', fontsize=15,labelpad=21)

ax.set_xticks(np.arange(1980,2018,1))

ax.tick_params(axis='both', which='major', labelsize=12)

ax.grid(False)

for item in ax.get_xticklabels():

    item.set_rotation(45)