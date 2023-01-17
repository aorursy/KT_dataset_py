# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings  
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
sales = pd.read_csv("../input/sales.csv")
navigation = pd.read_csv("../input/navigation.csv")
vimages = pd.read_csv("../input/vimages.csv")
train.head()
train.info()
train.describe()
train.nunique()
train.isna().sum()
sns.distplot(train.target)
sns.boxplot(train.target, orient='v')
plt.scatter(range(train.shape[0]), np.sort(train.target.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Quantity', fontsize=12)
f, axes = plt.subplots(2, 2, figsize=(15, 5))

for ii,i in enumerate(['product_type','product_gender']):
    sns.countplot(train[i], ax =axes[ii][0])
    sns.barplot(train[i], train.target, ax=axes[ii][1])
f, axes = plt.subplots(1,2, figsize=(20,10))
sns.countplot(y=train.macro_function, orient='v', ax=axes[0])
sns.barplot(y=train.macro_function, x=train.target, ax=axes[1])
f, axes = plt.subplots(1,2, figsize=(20,10))
sns.countplot(y=train.function, orient='v', ax=axes[0])
sns.barplot(y=train.function, x=train.target, ax=axes[1])
plt.figure(figsize=(15, 10))
sns.countplot(y=train.sub_function, orient='h')

train.model.value_counts(ascending=False).head(30)
f, axes = plt.subplots(1,2, figsize=(20,10))
sns.countplot(y=train.aesthetic_sub_line, orient='v', ax=axes[0])
sns.barplot(y=train.aesthetic_sub_line, x=train.target, ax=axes[1])
f, axes = plt.subplots(1,2, figsize=(20,10))
sns.countplot(y=train.macro_material, orient='v', ax=axes[0])
sns.barplot(y=train.macro_material, x=train.target, ax=axes[1])
train.color.value_counts(ascending=False).head(20)
plt.figure(figsize=(15, 10))
sns.distplot(train.fr_FR_price)
plt.figure(figsize=(15, 10))
sns.scatterplot(train.fr_FR_price, train.target)
plt.figure(figsize=(15,10))
sns.violinplot(x=train.product_type, y=train.target, inner="points")
plt.figure(figsize=(15,10))
sns.violinplot(x=train.product_gender, y=train.target, inner="points")
plt.figure(figsize=(15,10))
sns.violinplot(y=train.macro_function, x=train.target, inner="points")
sales.head()
sales.info()
sales.nunique()
sales.describe()
sales.isna().sum()
sales.country_number.value_counts(ascending=False).head()
sales.country_number.value_counts(ascending=False).head()
plt.figure(figsize=(15,5))
sns.distplot(sales.sales_quantity)
f, axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(sales.currency_rate_USD, ax=axes[0])
sns.scatterplot(sales.currency_rate_USD, sales.sales_quantity, ax=axes[1]) 
f, axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(sales.currency_rate_GBP, ax=axes[0])
sns.scatterplot(sales.currency_rate_GBP, sales.sales_quantity, ax=axes[1])
f, axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(sales.currency_rate_CNY, ax=axes[0])
sns.scatterplot(sales.currency_rate_CNY, sales.sales_quantity, ax=axes[1])
f, axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(sales.currency_rate_JPY, ax=axes[0])
sns.scatterplot(sales.currency_rate_JPY, sales.sales_quantity, ax=axes[1])
f, axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(sales.currency_rate_KRW, ax=axes[0])
sns.scatterplot(sales.currency_rate_KRW, sales.sales_quantity, ax=axes[1])
var = ['TotalBuzzPost', 'TotalBuzz', 'NetSentiment',
       'PositiveSentiment', 'NegativeSentiment', 'Impressions']

f, axes = plt.subplots(3,2, figsize=(15,5))

for ii, i in enumerate(var):
    axes = axes.flatten()
    sns.distplot(sales[i], ax =axes[ii]).set(xlabel=i)
f, axes = plt.subplots(3,2, figsize=(15,5))

for ii, i in enumerate(var):
    axes = axes.flatten()
    sns.scatterplot(sales[i], sales.sales_quantity, ax =axes[ii]).set(xlabel=i)
navigation.head()
navigation.info()
navigation.isna().sum()
navigation.nunique()
plt.figure(figsize=(15,5))
sns.distplot(navigation.page_views)
plt.figure(figsize=(15,5))
sns.boxplot(navigation.page_views)
f, axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(navigation.traffic_source, ax=axes[0])
sns.barplot(navigation.traffic_source, navigation.page_views, ax=axes[1])
navigation.addtocart.value_counts()
f, axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(navigation.day_visit, ax=axes[0])
sns.barplot(navigation.day_visit, navigation.page_views, ax=axes[1])
f, axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(navigation.month_visit, ax=axes[0])
sns.barplot(navigation.month_visit, navigation.page_views, ax=axes[1])
f, axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(navigation.website_version_zone_number, ax=axes[0])
sns.barplot(navigation.website_version_zone_number, navigation.page_views, ax=axes[1])
f, axes = plt.subplots(1,2, figsize=(15,5))
sns.countplot(navigation.website_version_country_number, ax=axes[0])
sns.barplot(navigation.website_version_country_number, navigation.page_views, ax=axes[1])