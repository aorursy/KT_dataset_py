# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting package

import seaborn as sns #plotting package

import warnings #to supress the warnings generated

warnings.filterwarnings('ignore')



import gc



color = sns.color_palette()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
PATH = '/kaggle/input/instacart-market-basket-analysis/'
#Reading all the data.

aisles = pd.read_csv(PATH + 'aisles.csv')

products = pd.read_csv(PATH + 'products.csv')

department = pd.read_csv(PATH + 'departments.csv')

orders = pd.read_csv(PATH + 'orders.csv')

train = pd.read_csv(PATH + 'order_products__train.csv')

#test = pd.read_csv(PATH + 'order_products__test.csv')

prior = pd.read_csv(PATH + 'order_products__prior.csv')
#Aisles data

aisles.head()
#number of aisles in the store

print('Shape of the dataset :',aisles.shape)

print('Number of unique aisles present: ', aisles.aisle_id.unique().shape[0])
#departments data

department.head()
#number of departments present in the store

print('Shape of the dataset :',department.shape)

print('Number of unique departments present: ', department.department_id.unique().shape[0])
#products dataset

products.head()
#number of products offered by the store.

print('Shape of the dataset :',products.shape)

print('Number of products :', products.product_id.unique().shape[0])
#merging table aisles and department with products.

products_merged = pd.merge(products, aisles, on='aisle_id', how='inner')

products_merged = pd.merge(products_merged, department, on='department_id', how='inner')



#shape of the new DataFrame.

products_merged.shape
#explore

products_merged.head()
#sorting by product_id (Optional)

products_merged.sort_values(by='product_id', inplace=True)

products_merged.reset_index(drop=True, inplace=True)
products_merged.head()
#department with most products.

products_merged.department.value_counts()
#aisles with most products.

products_merged.aisle.value_counts().head()
#getting the missing aisle and department subset.

missing = products_merged.loc[(products_merged.department == 'missing') & (products_merged.aisle == 'missing')]

missing.shape
#head()

missing.head(10)
#orders data

print("Shape of the dataset:", orders.shape)

orders.head()
#number of observatio,ns in each of the datasets (eval, train, prior)

plt.figure(figsize=(8, 5))

print('Number of observation in each set: \n{}'.format(orders.eval_set.value_counts()))

sns.countplot(x='eval_set', data=orders, color=color[0]);
#unique users of the instacart store.

print('Unique users of the store:', orders.user_id.unique().shape[0])
def unique_users_from_dataset(data):

    return data.unique().shape[0]



orders.groupby(by='eval_set')['user_id'].aggregate(unique_users_from_dataset)
#day of the week the orders were placed

plt.figure(figsize=(8, 5))

print('Frequency of orders on each day of the week: \n{}'.format(orders.order_dow.value_counts()))

sns.countplot('order_dow', data=orders, color=color[1])

plt.title('Frequency of orders on each day of the week')

plt.xlabel('DOW')

plt.ylabel('frequency of orders')
#hour of the day the orders were placed.

plt.figure(figsize=(8, 5))

print('Frequency of orders by hours in a day: \n{}'.format(orders.order_hour_of_day.value_counts().head())) # top 5 hours only

sns.countplot('order_hour_of_day', data=orders, color=color[2])

plt.title('Frequency of orders by hours in a day')

plt.xlabel('hours')

plt.ylabel('frequency of orders')
#number of NaN values in days_since_prior.

orders.days_since_prior_order.isnull().sum()
#extracting observations where this column has missing value.

missing = orders.loc[orders.days_since_prior_order.isnull()]

missing.head()
#checking assumption

missing.order_number.value_counts()
#imputing the missing values in days_since_prior_order with 0.

orders.days_since_prior_order.fillna(0, inplace=True)
#distribution of the days_since_prior_order column.

plt.figure(figsize=(10, 7))

sns.countplot('days_since_prior_order', data=orders, color=color[3])

plt.title('Number of days since previous order')

plt.xlabel('days')

plt.ylabel('frequency of days')

plt.xticks(rotation='vertical');
# creating a df which contains these two columns

grouped_df = orders.groupby(['order_dow', 'order_hour_of_day'])['order_id'].aggregate("count").reset_index()

grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_id')

plt.figure(figsize=(15, 6))

sns.heatmap(grouped_df, cmap='YlGn')

plt.xlabel('Hour of day')

plt.ylabel('Day of Week')

plt.title('Frequency of hour of day vs Day of week');
#exploring the mentioned dataset

prior.head()
train.head()
#Lets first see the number of unique orders present in both the datasets.

print('No. unique orders present in prior: {}'.format(prior.order_id.unique().shape[0]))

print('No. unique orders present in train: {}'.format(train.order_id.unique().shape[0]))
# Top 5 products ordered.

df = prior.product_id.value_counts().head()

df
#merging product information

prior_merged = pd.merge(prior, products_merged, on='product_id')

prior.shape, prior_merged.shape

gc.collect()
#now let's get the names top 5 ordered products.

prior_merged.product_name.value_counts().head()
#top 20

prior_merged.product_name.value_counts().head(20)
#Which aisle most of the ordered products belongs to

plt.figure(figsize=(9,5))

prior_merged.aisle.value_counts().head(20).plot.bar(color=color[4]);
#subsetting the data to get reordered items info.

reordered = prior_merged.loc[prior_merged.reordered == 1]

reordered.head()
#Top 5 most reordered.

reordered.product_name.value_counts().head(5)
#Let us first check the distribution of the departments.

distribution = prior_merged.department.value_counts()

labels = (np.array(distribution.index))

sizes = (np.array((distribution / prior_merged.shape[0])*100))



#Dataframe to old these values

dept_dist = pd.DataFrame()

dept_dist['Department'] = labels

dept_dist['distribution'] = sizes

dept_dist.head(21)
#Reordered ratio

reordered_ratio = prior_merged.groupby(by='department')['reordered'].agg(['mean']).reset_index()

reordered_ratio.sort_values(by='mean', ascending=False).head(21)
#plotting a graph.

plt.figure(figsize=(12,8))

sns.pointplot(reordered_ratio['department'].values, reordered_ratio['mean'].values, alpha=0.8, color=color[5])

plt.ylabel('Reorder ratio', fontsize=12)

plt.xlabel('Department', fontsize=12)

plt.title("Department wise reorder ratio", fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
#Let us first check the distribution of the aisles.

distribution = prior_merged.aisle.value_counts()

labels = (np.array(distribution.index))

sizes = (np.array((distribution / prior_merged.shape[0])*100))



#Dataframe to old these values

aisles_dist = pd.DataFrame()

aisles_dist['Aisle'] = labels

aisles_dist['distribution'] = sizes

aisles_dist.head(10) #showing only top 10 as there are 134 aisles.
#Reordered ratio for aisles

reordered_ratio = prior_merged.groupby(by='aisle')['reordered'].agg(['mean']).reset_index()

top = reordered_ratio.sort_values(by='mean', ascending=False).head(10) # Top 10

bottom = reordered_ratio.sort_values(by='mean').head(10) # Bottom 10



#plotting a graph.

plt.figure(figsize=(15,8))

sns.pointplot(top['aisle'].values, top['mean'].values, alpha=0.8, color=color[6])

#sns.pointplot(bottom['aisle'].values, bottom['mean'].values, alpha=0.8, color=color[6])

plt.ylabel('Reorder ratio', fontsize=12)

plt.xlabel('Aisle', fontsize=12)

plt.title("Aisle wise reorder ratio", fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
#merging the products information with train.

train_merged = pd.merge(train, products, on='product_id', how='inner')

print(train.shape, train_merged.shape)
#merging the orders info with train and prior

train_merged = pd.merge(train_merged, orders, on='order_id', how='left')

print(train.shape, train_merged.shape)
#merging the orders info with prior

prior_merged = pd.merge(prior_merged, orders, on='order_id', how='left')

print(prior.shape, prior_merged.shape)
train_merged.head()
prior_merged.head()
#getting the reordered subset

reordered_ratio = train_merged.groupby(["order_dow"])["reordered"].aggregate("mean").reset_index()



plt.figure(figsize=(10,5))

sns.barplot(reordered_ratio['order_dow'].values, reordered_ratio['reordered'].values, alpha=0.8, color=color[8])

plt.ylabel('Reorder ratio', fontsize=12)

plt.xlabel('Day of week', fontsize=12)

plt.title("Reorder ratio across day of week", fontsize=15)

plt.xticks(rotation='vertical')

plt.ylim(0.5, 0.7)

plt.show()
#getting the reordered subset

reordered_ratio = train_merged.groupby(["order_hour_of_day"])["reordered"].aggregate("mean").reset_index()



plt.figure(figsize=(12,6))

sns.barplot(reordered_ratio['order_hour_of_day'].values, reordered_ratio['reordered'].values, alpha=0.8, color=color[9])

plt.ylabel('Reorder ratio', fontsize=12)

plt.xlabel('Hour of day', fontsize=12)

plt.title("Reorder ratio across hour of day", fontsize=15)

plt.xticks(rotation='vertical')

plt.ylim(0.5, 0.7)

plt.show()