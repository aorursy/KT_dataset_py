## Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
## Importing the dataset

dataset = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
dataset.head(10)
dataset.info()
# Splitting the comma separated values in the country column

ds1 = dataset.assign(country= dataset['country'].str.split(', ')).explode('country')

ds1
# Applying filter to verify the split columns fetch correct rows 

ds1[ds1['show_id']==81145628]
## Visualising the dataset with type and year of release

plt.figure(figsize=(15,8))

plt.style.use('dark_background')

sns.countplot(x='type',data=dataset,hue='release_year')

plt.xlabel('Type- Movie or TV show',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.legend(loc='best',ncol=3)

plt.show()
## Visualising the dataset with type and selected countries

plt.figure(figsize=(15,12))

myCountries=['India','France','United States','United Kingdom','Germany','Australia','Brazil','China','Singapore','Russia',

            'Canada','Spain','Denmark','South Korea','Mexico']

df = ds1[ds1.country.isin(myCountries)]

ax = sns.countplot(x='type',data=df,hue='country')

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.0f}'.format(height), (x + 0.02, y + height+ 15),rotation=90,size=12)

plt.xlabel('Type- Movie or TV show',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.legend(loc='best',ncol=1,fontsize=12)

plt.show()
# Applying filter on dataframe to cross check the number in the graph above

df[(df['country'].isin(['United States'])) & (df['type']=='Movie')]
## Visualising the dataset with type for the past 10 years

plt.figure(figsize=(12,10))

yearlist=list(range(2010,2021,1))

recentYears = dataset[dataset['release_year'].isin(yearlist)]

ax = sns.countplot(x='type',data=recentYears,hue=recentYears['release_year'])

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.0f}'.format(height), (x + 0.02, y + height+ 8),rotation=90,size=12)

plt.xlabel('Type- Movie or TV show',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.legend(loc='best',ncol=1)

plt.show()
# Splitting the comma separated values in the listed_in column

ds2 = ds1.assign(listed_in= ds1['listed_in'].str.split(', ')).explode('listed_in')

ds2
# Removing the duplicate rows based on show_id after the explode operation 

ds2 = ds2.drop_duplicates(subset=None, keep='first', inplace=False)
ds2.head(10)
# Visualising the dataset with movie content available in selected countries

fig, axes = plt.subplots(15,1,figsize=(12,100),squeeze=False)

for i in range(0,15):

    plt.tight_layout(pad=5)

    movieOnly = ds2[(ds2['type']=='Movie') & (ds2['country'].str.contains(myCountries[i]))]

    sc = sns.countplot(x='listed_in',data=movieOnly,ax=axes[i][0],order = movieOnly['listed_in'].value_counts().index)

    axes[i][0].set_title('Movies in ' + myCountries[i],fontweight='bold')

    axes[i][0].set_xlabel('Genre of Netflix movies')

    axes[i][0].set_xticklabels(movieOnly['listed_in'].unique(),rotation=90)

    for p in sc.patches:

        width, height = p.get_width(), p.get_height()

        x, y = p.get_xy() 

        sc.annotate('{:.0f}'.format(height), (x+0.2, y + height))

plt.show()
# Visualising the dataset with TV content available in selected countries



fig, axes = plt.subplots(15,1,figsize=(12,100),squeeze=False)

for i in range(0,15):

    plt.tight_layout(pad=2)

    TVOnly = ds2[(ds2['type']=='TV Show') & (ds2['country']==myCountries[i])]

    sc = sns.countplot(x='listed_in',data=TVOnly,ax=axes[i][0],order = TVOnly['listed_in'].value_counts().index)

    axes[i][0].set_title('TV Shows in ' + myCountries[i],fontweight='bold')

    axes[i][0].set_xlabel('Genre of Netflix TV shows')

    axes[i][0].set_xticklabels(TVOnly['listed_in'].unique(),rotation=90)

    for p in sc.patches:

        width, height = p.get_width(), p.get_height()

        x, y = p.get_xy() 

        sc.annotate('{:.0f}'.format(height), (x+0.2, y + height))

plt.show()
# Formatting the date_added column

ds1['date_added'] = pd.to_datetime(ds1["date_added"])
ds1
# Visualising the number of movies added in 2019.

df = ds1[(ds1['release_year']==2019) & (ds1['type']=='Movie')]

df['month'] = df['date_added'].dt.month

sns.countplot(x=df['month'],data=df)

plt.xlabel('Months-2019')

plt.ylabel('Number of movies')

plt.title('Number of movies added monthwise in 2019')

plt.show()
# Visualising the number of TV shows added in 2019.

df = ds1[(ds1['release_year']==2019) & (ds1['type']=='TV Show')]

df['month'] = df['date_added'].dt.month

sns.countplot(x=df['month'],data=df)

plt.xlabel('Months-2019')

plt.ylabel('Number of TV shows')

plt.title('Number of TV shows added monthwise in 2019')

plt.show()