# IMPORTS

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
apps = pd.read_csv('../input/googleplaystore.csv')
apps.head()
apps.info()
len(apps['App'].unique())
apps.groupby('App').count()[apps.groupby('App').count()['Category'] != 1]['Category'].tail()
apps[apps['App'] == 'theScore: Live Sports Scores, News, Stats & Videos']
apps_clean = apps.drop_duplicates(subset='App')

apps_clean.shape
(apps_clean['Category'].unique())
apps_clean[apps_clean['Category'] == "1.9"]
apps_clean = apps_clean.drop(index=10472)
assert len(apps_clean[apps_clean['Category'] == "1.9"]) == 0
apps_clean.describe()
apps_clean.Rating.isnull().sum()
apps_clean.Reviews = pd.to_numeric(apps_clean.Reviews)
apps_clean.Reviews.describe()
def size_converter(data):

    if data[-1] == 'M':

        return float(data[:-1])

    elif data[-1] == 'k':

        return float(data[:-1]) / 1024

    else:

        return 0

    
apps_clean['Size'] = apps_clean['Size'].apply(size_converter)
apps_clean['Size'].describe()
apps_clean[apps_clean['Size'] == 100 ].groupby('Category')['Size'].count().sort_values(ascending = False)
apps_clean['Installs'].unique()
def installs(data):

    x = data.split('+')[0]

    try:

        if int(x) <= 100:

            return '100-'

        else:

             return data

    except ValueError:

        return data
apps_clean['Installs'] = apps_clean['Installs'].apply(installs)
apps_clean['Type'].unique()
apps_clean[apps_clean['Type'].isnull()]
apps_clean.drop(index=9148, inplace=True)
apps_clean['Price'].unique()
def price_converter(price):

    try:

        return float(price.split('$')[1])

    except IndexError:

        return float(price)
apps_clean['Price'] = apps_clean['Price'].apply(price_converter)
apps_clean['Price'].describe()
apps_clean[apps_clean['Price'] == 400.0]
apps_clean[apps_clean['Price'] > 100].head()
apps_clean['Content Rating'].unique()
apps_clean['Genres'].nunique()
apps_clean[['Category', 'Genres']][:15]
apps_clean.drop('Genres', axis=1, inplace=True)
apps_clean['Last Updated'].describe()
apps_clean['Last Updated'] = pd.to_datetime(apps_clean['Last Updated'])
apps_clean.info()
apps_clean['Current Ver'].nunique()
apps_clean['Android Ver'].unique()
apps_clean.drop(['Current Ver', 'Android Ver'], axis=1, inplace=True)
apps_clean.head()
plt.rcParams["figure.figsize"] = (10, 5)
apps_clean['Rating'].plot(kind='hist');

plt.title('Rating Distribution')

plt.xlabel('Rating(in stars)')
sns.boxplot(y='Rating', x='Category',data=apps_clean)

plt.tight_layout()

plt.xticks(rotation='vertical')



plt.axhline(apps_clean.Rating.median(), color='red')
fig, axes = plt.subplots(nrows=1, ncols=2, sharex='col')

apps_clean[apps_clean['Category'] == 'ART_AND_DESIGN']['Rating'].plot(kind='hist', ax=axes[0])

apps_clean[apps_clean['Category'] == 'TOOLS']['Rating'].plot(kind='hist', ax=axes[1])



axes[0].title.set_text('ART_AND_DESIGN')

axes[1].title.set_text('TOOLS')
apps_clean[apps_clean['Category'] == 'ART_AND_DESIGN']['App'][10:20]
apps_clean[apps_clean['Category'] == 'TOOLS']['App'][10:20]
apps_clean.groupby('Category')['Rating'].mean().sort_values()
apps_clean['Last Updated'].describe()
apps_clean[apps_clean['Last Updated'] == '2010-05-21 00:00:00']