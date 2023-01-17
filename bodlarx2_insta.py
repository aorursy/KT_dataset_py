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
#importing required libraries
import pandas as pd
import numpy as np
import glob
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.getcwd()
os.chdir("../input")
#Finding the list of files to uplod
pattern = '*.csv'
csv_files = glob.glob(pattern)
print(csv_files)
#uploading files
aisles = pd.read_csv("aisles.csv", dtype={'aisle_id': 'object'})
departments = pd.read_csv("departments.csv", dtype={'department_id':'object'})
orders = pd.read_csv("orders.csv", dtype = {'order_number': 'uint32',
                                            'order_dow': 'uint32',
                                            'order_hour_of_day': 'uint32',
                                            'days_since_prior_order': 'float16',
                                            'order_id': 'object',
                                            'user_id': 'object'})
orders_products_prior = pd.read_csv("order_products__prior.csv", dtype={'add_to_cart_order':'uint32',
                                                                        'reordered': 'uint32',
                                                                        'order_id': 'object', 
                                                                        'product_id': 'object'})
orders_products_train = pd.read_csv("order_products__train.csv", dtype={'add_to_cart_order':'uint32',
                                                                        'reordered': 'uint32',
                                                                        'order_id': 'object', 
                                                                        'product_id': 'object'})
products = pd.read_csv("products.csv", dtype={'product_id': 'object',
                                              'aisle_id': 'object', 
                                              'department_id': 'object'})
orders.dtypes
# Assigining names to the datasets
data_list = [aisles, departments, orders, orders_products_prior, orders_products_train, products]
data_list_name = ['aisles', 'departments', 'orders', 'orders_products_prior', 'orders_products_train', 'products']
j=0
for i in data_list:
    i.name = data_list_name[j]
    j += 1
def null_columns(x):
    y = x.columns[x.isnull().any()]
    return y

for dataset in data_list:
    if len(null_columns(dataset)) == 0:
        print('Dataset ' + dataset.name + ' has no null values ')
    else:
        print('Dataset '+ dataset.name + ' has null values in column '+ str([i for i in null_columns(dataset)]))
#Dataset orders has null values in column ['days_since_prior_order'] 
#Finding the number of null values in orders.days_since_prior_order

sum(orders.days_since_prior_order.isnull())

#There are 206209 null values
#Lets examine the null values
orders.loc[orders.days_since_prior_order.isnull()==True,['user_id', 'order_number']]
#days_since_prior_order attribue for first order for all the users is null so it can be changed to 0

orders.days_since_prior_order = orders.days_since_prior_order.fillna(int(0))
# Merging Orders and orders_products_prior
total = orders.merge(orders_products_prior, how = 'left', on='order_id').merge(orders_products_train,
                                                                              how='left', on ='order_id')
total.head()
#for i in ('product_id', 'add_to_cart_order', 'reordered'):
total.product_id_x = total.product_id_x.fillna(total.product_id_y)
total.add_to_cart_order_x = total.add_to_cart_order_x.fillna(total.add_to_cart_order_y)
total.reordered_x = total.reordered_x.fillna(total.reordered_y)
#Dropping the duplicate columns
total = total.drop(['product_id_y', 'add_to_cart_order_y', 'reordered_y'], axis=1)
total.head()
#Renaming the columns after merging
total=total.rename(columns = {'product_id_x':'product_id',
                             'add_to_cart_order_x': 'add_to_cart_order',
                             'reordered_x': 'reordered'})
total.head()
total.dtypes
#Changing the data type of 'add_to_cart_order','reordered' to float16 to improve performance
total[['add_to_cart_order','reordered']] = total[['add_to_cart_order','reordered']].astype('float16')
total.dtypes
aisles.head()
products.head()
departments.head()
#Merging total with products dataframe as total1 
total1 = total.merge(products, on='product_id', how = 'left')
#Merging total1 and aisles
total1 = total1.merge(aisles, on='aisle_id', how = 'left')
#Merging total1 and departments
total1 = total1.merge(departments, on='department_id', how = 'left')
os.getcwd()
total1.to_csv("/kaggle/working/total1.csv", index=None)
os.listdir('/kaggle')
total1.head()
total1.shape
total1.columns
#Number of orders per day of the week
plt.figure(figsize=(13,9))
sns.countplot(x='order_dow', data=total1, )
ax = plt.gca()
ax.set_xticklabels(['Sunday','Monday','Tuesday','Wednesday','Thursday', 'Friday','Saturday'])
plt.xticks(fontsize=14)
plt.xlabel('Day of the Week', fontsize=14)
plt.ylabel('Number of Orders', fontsize=14)
plt.title('Number of orders Vs Day of the Week', fontsize=14)
plt.show()
#Number of orders per hour of the day
plt.figure(figsize=(13,9))
sns.countplot(x='order_hour_of_day', data=total1)
plt.xlabel('Hour of the Day', fontsize=14)
plt.ylabel('Number of Orders', fontsize=14)
plt.xticks(fontsize=14)
plt.title('Number of orders Vs Hour of the Day', fontsize=14)
plt.show()
#Top 10 Aisles
aisle_top10 = pd.DataFrame(total1.groupby('aisle').aisle.count().sort_values(ascending=False).head(10))
plt.figure(figsize=(13,9))
sns.barplot(aisle_top10.index, aisle_top10.aisle, color='olive').set_title('Top 10 Aisles', fontsize=14)
plt.xticks(rotation=90, fontsize=14)
plt.xlabel('Aisle', fontsize=14)
plt.ylabel('Number of Orders', fontsize=14)
plt.show()
#Top 10 Departments
departments_top10 = pd.DataFrame(total1.groupby('department').department.count().sort_values(ascending=False).head(10))
plt.figure(figsize=(13,9))
sns.barplot(departments_top10.index, departments_top10.department, color='limegreen').set_title('Top 10 Departments', fontsize=14)
plt.xticks(fontsize=14, rotation=20)
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.xlabel('Department', fontsize=14)
plt.ylabel('Number of Orders', fontsize=14)
plt.show()
#Top 10 Products
products_top10 = pd.DataFrame(total1.groupby('product_name').product_name.count().sort_values(ascending=False).head(10))
plt.figure(figsize=(13,9))
sns.barplot(products_top10.index, products_top10.product_name, color='teal').set_title('Top 10 Products', fontsize=14)
plt.xticks(rotation=90, fontsize=14)
plt.xlabel('Product', fontsize=14)
plt.ylabel('Number of Orders', fontsize=14)
plt.show()
#Top 20 reordered Products
reordered = pd.DataFrame(total1.loc[total1['reordered']==1.0].groupby('product_name').size().sort_values(ascending=False).head(20).reset_index(name='counts'))
plt.figure(figsize=(13,9))
sns.barplot(reordered.product_name, reordered.counts, color='darkgoldenrod')
plt.xticks(rotation=90, fontsize=14)
plt.xlabel('Product', fontsize=14)
plt.ylabel('Number of Orders', fontsize=14)
plt.title('Top 20 reordered products', fontsize=14)
plt.show()
#Added to the Cart sequence Vs reordered
df2 = total1.groupby(['add_to_cart_order', 'reordered']).size().to_frame('Counts')
df = df2.groupby(level=[0]).apply(lambda x: round(x / x.sum()*100,2)).reset_index()
plt.figure(figsize=(25,9))
sns.barplot(df.add_to_cart_order, df.Counts, hue=df.reordered)
plt.xlim(xmax=50)
plt.xlabel('Order of adding to the Cart', fontsize=14)
plt.ylabel('Percentage',fontsize=14)
plt.title('Order of adding to the Cart VS reordered',fontsize=14)
plt.show()
#Department Vs Reordered
dep_reord = total1.groupby(['department', 'reordered']).size().to_frame('Counts')
dep_reord1 = dep_reord.groupby(level=[0]).apply(lambda x: round(x / x.sum()*100,2)).reset_index()
plt.figure(figsize=(25,9))
sns.barplot(dep_reord1.department, dep_reord1.Counts, hue=dep_reord1.reordered)
plt.xlabel('Department', fontsize=14)
plt.ylabel('Percentage',fontsize=14)
plt.title('Department VS reordered',fontsize=14)
plt.show()
#Top 20 Users
top_20_users = total1.groupby('user_id').size().sort_values(ascending=False).to_frame('Counts').head(20)
plt.figure(figsize=(25,9))
sns.barplot(top_20_users.index,top_20_users.Counts, color='limegreen')
plt.xticks(rotation=45)
plt.xlabel('User')
plt.ylabel('Number of Items Purchased')
plt.title('Top 20 Users')
plt.show()
#Organic VS reordered
df4 = total1[['product_name','reordered']]
df4['is_organic'] = df4['product_name'].str.contains('Organic')
organic2 = df4.groupby(['is_organic','reordered']).size().to_frame('Counts')
organic3 = organic2.groupby(level=[0]).apply(lambda x: round(x / x.sum()*100,2)).reset_index()
plt.figure(figsize=(13,9))
sns.barplot(organic3.is_organic, organic3.Counts, hue=organic3.reordered)
plt.xlabel('Is_Organic', fontsize=14)
plt.ylabel('Percentage',fontsize=14)
plt.title('Organic VS reordered',fontsize=14)
plt.show()
#Avg add_to_cart for each product
Avg_order = total1.groupby(['product_name'])['add_to_cart_order'].aggregate('mean').to_frame('Avg_add_to_cart')
Avg_order['Avg_add_to_cart'] = Avg_order['Avg_add_to_cart'].apply(lambda x: round(x,2))
Avg_order.sort_values(by='Avg_add_to_cart')
#Gluten free VS reordered
df5 = total1[['product_name','reordered']]
df5['is_gluten_free'] = df4['product_name'].str.contains('gluten' and 'free')
gluten_free = df5.groupby(['is_gluten_free','reordered']).size().to_frame('Counts')
gluten_free1 = gluten_free.groupby(level=[0]).apply(lambda x: round(x / x.sum()*100,2)).reset_index()
plt.figure(figsize=(13,9))
sns.barplot(gluten_free1.is_gluten_free, gluten_free1.Counts, hue=gluten_free1.reordered)
plt.xlabel('is_gluten_free', fontsize=14)
plt.ylabel('reorder_Percentage',fontsize=14)
plt.title('Gluten Free VS reordered',fontsize=14)
plt.show()
#Asian VS reordered
df6 = total1[['product_name','reordered']]
df6['is_asian'] = df4['product_name'].str.contains('asian')
asian = df6.groupby(['is_asian','reordered']).size().to_frame('Counts')
asian1 = asian.groupby(level=[0]).apply(lambda x: round(x / x.sum()*100,2)).reset_index()
plt.figure(figsize=(13,9))
sns.barplot(asian1.is_asian, asian1.Counts, hue=asian1.reordered)
plt.xlabel('is_asian', fontsize=14)
plt.ylabel('reorder_Percentage',fontsize=14)
plt.title('Asian food VS reordered',fontsize=14)
plt.show()
#Low fat VS reordered
df11 = total1[['product_name','reordered']]
df11['low_fat'] = df4['product_name'].str.contains('low' and 'fat')
low_fat = df11.groupby(['low_fat','reordered']).size().to_frame('Counts')
low_fat1 = low_fat.groupby(level=[0]).apply(lambda x: round(x / x.sum()*100,2)).reset_index()
plt.figure(figsize=(13,9))
sns.barplot(low_fat1.low_fat, low_fat1.Counts, low_fat1.reordered)
plt.xlabel('low_fat', fontsize=14)
plt.ylabel('reorder_Percentage',fontsize=14)
plt.title('low_fat VS reordered',fontsize=14)
plt.show()
def timezone(i):
    if i < 3:
        return 'midnight'
    elif i < 6:
        return 'early morning'
    elif i < 12:
        return 'morning'
    elif i < 18:
        return 'noon'
    else:
        return 'night'
    
df7 = total1[['department', 'order_hour_of_day']]
df7['time_zone'] = df7.order_hour_of_day.map(timezone)
time_zone = df7.groupby(['department', 'time_zone']).size().to_frame('Counts')
timezone1 = time_zone['Counts'].groupby(level=0, group_keys=False)
timezone2 = timezone1.apply(lambda x: x.sort_values(ascending=False).head(1)).to_frame('Counts')
#timezone2 = timezone2.reset_index()
timezone2
total1.head(10).T

