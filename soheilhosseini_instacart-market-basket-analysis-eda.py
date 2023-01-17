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
import zipfile



zf = zipfile.ZipFile('../input/instacart-market-basket-analysis/order_products__train.csv.zip') 

order_products__train = pd.read_csv(zf.open('order_products__train.csv'))



zf = zipfile.ZipFile('../input/instacart-market-basket-analysis/order_products__prior.csv.zip') 

order_products__prior = pd.read_csv(zf.open('order_products__prior.csv'))



zf = zipfile.ZipFile('../input/instacart-market-basket-analysis/sample_submission.csv.zip') 

sample_submission = pd.read_csv(zf.open('sample_submission.csv'))



zf = zipfile.ZipFile('../input/instacart-market-basket-analysis/aisles.csv.zip') 

aisles = pd.read_csv(zf.open('aisles.csv'))



zf = zipfile.ZipFile('../input/instacart-market-basket-analysis/orders.csv.zip') 

orders = pd.read_csv(zf.open('orders.csv'))



zf = zipfile.ZipFile('../input/instacart-market-basket-analysis/products.csv.zip') 

products = pd.read_csv(zf.open('products.csv'))



zf = zipfile.ZipFile('../input/instacart-market-basket-analysis/departments.csv.zip') 

departments = pd.read_csv(zf.open('departments.csv'))

orders.info()
order_products__train.info()
#Checking memory usage

order_products__prior.info()
order_products__prior.describe()
#Reducing memory usage



order_products__prior = order_products__prior.astype('Int32')

order_products__prior.info()
import seaborn as sns

import matplotlib.pyplot as plt
order_products__train.head()
pd.options.display.max_rows = 999

orders.loc[orders['eval_set']=='train'].sort_values(by='order_id').head()
order_products__train_dow = order_products__train.merge(orders, on='order_id')

plt.grid(b=True)



plt.title('Day of week on being reordered')

C= order_products__train_dow[['reordered','order_dow']].groupby('order_dow').mean()

sns.barplot(x=C.index[0:70], y=C['reordered'])

plt.ylim(0,1)

plt.show()
aisles.head()
orders.eval_set.unique()
#How many prior orders are available based on order_numer?

plt.figure(figsize=(27,5))

sns.countplot(x='order_number', data = orders.loc[orders.eval_set=='prior'], palette='plasma')

plt.show()
np.sum(pd.isnull(orders))
orders.fillna(0, inplace=True)
orders.head()
orders.user_id.nunique()
# Effect of hour of day on order volume

import matplotlib.pyplot as plt 

sns.countplot(x='order_hour_of_day', data =orders)

plt.show()
# Effect of day of the w on order volume

sns.countplot(x='order_dow', data =orders)

plt.show()
plt.figure(figsize=(2,5))

sns.countplot(orders['eval_set'], palette='plasma')

plt.ticklabel_format(style='plain', axis='y')

plt.show()
products.head()
order_products__train.head()
order_products__train_merged = order_products__train.merge(products, on='product_id')
order_products__train_merged.head()
#List of best selling products

best_seller_products = order_products__train_merged.product_name.value_counts()[0:10]

best_seller_products
plt.figure(figsize=(8,6))

sns.barplot(x= best_seller_products.index, y=best_seller_products )

plt.xticks(rotation=90)

plt.show()
order_products__train.reordered.value_counts()
order_products__train.reordered.value_counts(normalize=True)
# How many times customers reordered?

plt.figure(figsize=(2.5,4))

plt.title('How many times customers reordered?')

sns.countplot(x='reordered', data= order_products__train, palette='plasma')

plt.show()
pd.options.display.max_rows = 999

order_products__train_merged.sample(5)
# Organic vs Not-organic products

order_products__train_merged['Organic or Not'] = order_products__train_merged.product_name.str.lower().str.contains('organic')

order_products__train_merged['Organic or Not'].replace({True:'Organic', False:'Not-Organic'}, inplace=True)
order_products__train_merged.head()
order_products__train_merged['Organic or Not'].value_counts()
plt.figure(figsize=(2.5,4))

sns.countplot(order_products__train_merged['Organic or Not'], palette='plasma')

plt.ticklabel_format(style='plain', axis='y')

plt.xlabel('')

plt.show()
orders.head(15)
plt.figure(figsize=(10,5))

sns.countplot(orders['days_since_prior_order'])



plt.title('distribution of customer orders since prior order'.upper())

plt.xticks(rotation=90)



plt.show()
products.head()
aisles.head()
product_aisle = products.merge(aisles, on='aisle_id')

product_aisle.head()
order_products__prior_aisle = order_products__prior.merge(product_aisle, on='product_id')

order_products__prior_aisle.head()
A = order_products__prior_aisle[['aisle_id','aisle']].groupby(by='aisle').count().sort_values(by='aisle_id', ascending=False)

plt.figure(figsize=(10,5))

plt.title('what are peoples favorite Aisles?')



sns.barplot(x=A.index[0:20], y=A['aisle_id'][0:20], color='red')

plt.ticklabel_format(style='plain', axis='y')



plt.xticks(rotation=90)



plt.show()
order_products__prior_aisle.head()
order_products__prior_aisle_dep = order_products__prior_aisle.merge(departments, on='department_id')

order_products__prior_aisle_dep.head()
B= order_products__prior_aisle_dep[['reordered','department']].groupby('department').mean()

plt.plot(B.index, B['reordered'], 'r-o', linewidth=3, markersize=10)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,6))

plt.grid(b=True)

plt.title('Effect of add to cart order on being reordered')

C= order_products__prior_aisle_dep[['reordered','add_to_cart_order']].groupby('add_to_cart_order').mean()

plt.plot(C.index[0:70], C['reordered'][0:70], color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=12)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(10,5))

plt.title('Distribution in departments?')



sns.countplot(x=order_products__prior_aisle_dep['department'], color='red')

plt.ticklabel_format(style='plain', axis='y')



plt.xticks(rotation=90)



plt.show()
order_products__prior_aisle_dep['department'].value_counts()
order_products__prior_aisle_dep['department'].value_counts().plot.pie( figsize=(10, 10), autopct='%1.1f%%')

plt.show()