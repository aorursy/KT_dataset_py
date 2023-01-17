# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
playstore = pd.read_csv('../input/googleplaystore.csv')
reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')
playstore.head()
playstore.isnull().sum()
reviews.head()
playstore.shape
playstore.dtypes
def get_reviews(reviews):
    if reviews.endswith('.0M'):
        reviews = reviews[:-3] + '000000'
        return int(reviews)
    else:
        return int(reviews)
playstore['Reviews'] = playstore['Reviews'].apply(get_reviews)
def get_size(size):
    if size == 'Varies with device':
        return 10
    elif size.endswith('M'):
        size = size[:-1]
        return float(size)
playstore['Size'] = playstore['Size'].apply(get_size)
x = np.unique(playstore['Installs'])
x
def get_installs(installs):
    installs = re.sub(',', '', installs)
    if installs.endswith('+'):
        installs = installs[:-1]
        return int(installs)
    else:
        return 0
playstore['Installs'] = playstore['Installs'].apply(get_installs)
p = np.unique(playstore['Price'])
p
def get_price(price):
    price = re.sub('\$', '', price)
    if price == 'Everyone':
        return float(0)
    else:
        return float(price)
playstore['Price'] = playstore['Price'].apply(get_price)
mean_rating = playstore['Rating'].mean()
mean_size = playstore['Size'].mean()
playstore['Rating'] = playstore['Rating'].fillna(mean_rating)
playstore['Size'] = playstore['Size'].fillna(mean_size)
playstore.loc[playstore['Type'].isnull()]
playstore['Type'] = playstore['Type'].fillna('Free')
playstore.loc[playstore['Content Rating'].isnull()]
playstore['Content Rating'] = playstore['Content Rating'].fillna('Everyone')
playstore.dropna(how = 'any', inplace = True)
playstore.isnull().sum()
cat = np.unique(playstore['Category'])
cat
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Category','Installs', data = playstore)
plt.show()
health = playstore.loc[playstore['Category'] == 'HEALTH_AND_FITNESS']
health
health['Rating'].mean()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.countplot('Category', data = playstore)
plt.show()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Category','Rating', data = playstore)
plt.show()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Category','Reviews', data = playstore)
plt.show()
plt.subplots(figsize = (8,4))
sns.distplot(playstore['Size'], bins = 15)
plt.show()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Category','Size', data = playstore)
plt.show()
plt.subplots(figsize = (8,4))
plt.xticks(rotation = 90)
sns.barplot('Type','Installs', data = playstore)
plt.show()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Category','Installs',hue = 'Type', data = playstore)
plt.show()
plt.subplots(figsize = (8,4))
sns.distplot(playstore['Price'], bins = 2)
plt.show()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Content Rating','Installs', data = playstore)
plt.show()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Content Rating', 'Reviews', data = playstore)
plt.show()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.countplot('Content Rating', data = playstore)
plt.show()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Rating', 'Reviews', data = playstore)
plt.show()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.countplot('Rating', data = playstore)
plt.show()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Android Ver', 'Rating', data = playstore)
plt.show()
plt.subplots(figsize = (18,8))
plt.xticks(rotation = 90)
sns.barplot('Android Ver', 'Reviews', data = playstore)
plt.show()