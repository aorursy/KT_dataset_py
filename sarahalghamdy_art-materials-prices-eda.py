import seaborn as sns

from wordcloud import WordCloud

import re

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('seaborn-white') 

%matplotlib inline 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
result_ = pd.read_csv('/kaggle/input/art-materials-prices/Art_Materials.csv')
result_.head()
result_.drop('Unnamed: 0',axis =1,inplace=True)
# Find the average price for each brand

price_avg = pd.DataFrame(result_.groupby('brand')['price'].mean()).sort_values(

    by= ['price'],ascending= False).dropna().reset_index()
price_avg.head()
fig, ax = plt.subplots(figsize=(20, 6))

chart = sns.barplot(x='brand',

        y='price',

        hue=None,

        data=price_avg,

                ax = ax, palette= "PuOr_r");

chart.set_xticklabels(chart.get_xticklabels(), rotation=90);

plt.title('The Average Price For Each Brand');
# Find the average rating for each brand

rating_avg = pd.DataFrame(result_.groupby(['brand',])['rating','price'].mean()).sort_values(

    by= ['rating'],ascending= False).reset_index()
rating_avg.head()
#  Some brands weren't rated at all

rating_avg.tail()




# Rating avrage for each brand



fig, ax = plt.subplots(figsize=(20, 6))

chart = sns.barplot(x='brand',

        y='rating',

        hue=None,

        data=rating_avg,

                ax = ax ,  palette = 'Purples_r' );

chart.set_xticklabels(chart.get_xticklabels(), rotation=90);

plt.title('The Average Rating For Each Brand');
# More than the average number of products per brand

brand_count = pd.DataFrame(result_.groupby('brand')['product_name'].count()).reset_index()

brand_count_mean = brand_count[brand_count.product_name > brand_count.product_name.mean()]
brand_count.head()
brand_count_mean.head()
#  number of products is more than the average number of products per brand

fig, ax = plt.subplots(figsize=(20, 6))

chart = sns.barplot(x='brand',

        y='product_name',

        hue=None,

        data=brand_count_mean.sort_values(by = 'product_name',ascending = False),

                ax = ax , palette = 'Purples_r' );

chart.set_xticklabels(chart.get_xticklabels(), rotation=70);

plt.title('More than the average number of products per brand');
#  Count the products that has higher price than 100 for each brand



high_price = result_[result_['price']>100].sort_values(by = ['price'], ascending= False)

high_price.head()
#  Count the products that have higher price than 100 for each brand

chart = sns.catplot(

    data= high_price,

    x='price',

    kind='count',

    palette='Set1',

    row='brand',

    aspect=4,

    height=4

)

chart.set_xticklabels(rotation=65, horizontalalignment='right');
# Distribution of prices

plt.figure(figsize=(15,8))

sns.distplot(result_['price'][result_['price'].notnull()]);

plt.title('Distribution Of Price');



#  Distribution of rate

plt.figure(figsize=(15,8))

sns.distplot(result_['rating'][result_['rating'].notnull()])

plt.title('Distribution Of Rating');