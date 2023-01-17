# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
playstore = pd.read_csv('../input/googleplaystore.csv', header="infer")
playstore.head()
playstore.describe()
playstore.info()
playstore.isnull().any()
playstore.dropna(inplace=True)
import re

def fixPrice(column):
    return re.sub('([a-zA-Z]+|\$)', '', column)

# Convert price to float. 
playstore['Price'] = playstore['Price'].apply(fixPrice)
playstore['Price'] = pd.to_numeric(playstore['Price'])

# Get most expensive app.
max_val = playstore['Price'].max()
playstore[playstore['Price'] == max_val]
len(playstore['Category'].unique())
from pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

top_five_categories = playstore['Category'].value_counts().head().rename_axis('categories').reset_index(name='counts')
g = sns.barplot(top_five_categories.categories, top_five_categories.counts)
g = sns.color_palette("husl", 3) 
plt.title("Top 5 categories by number of apps") 

# Adjusting figure's size
rcParams['figure.figsize'] = 5, 10
plt.show(g)
playstore['Type'].value_counts()
category_rating = playstore.groupby('Category')['Rating'].mean().sort_values(ascending=False)
category_rating
import seaborn as sns

sns.set()
plt.rcParams['figure.figsize'] = [25, 10]
sns.barplot(x=category_rating[:10].index, y=category_rating[:10].get_values())
paid_mean_price =  playstore[playstore['Type'] == 'Paid'].groupby('Category')['Price'].mean().sort_values(ascending=False)
paid_mean_price
sns.set()
sns.barplot(x=paid_mean_price[:10].index, y=paid_mean_price[:10].get_values())
pivoted = playstore.pivot_table(index='Category', columns='Type', values='Rating', aggfunc='mean')
pivoted
pivoted.mean()
