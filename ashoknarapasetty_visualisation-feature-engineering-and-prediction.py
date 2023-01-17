import os
import glob
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir('../input')
os.getcwd()
for i in glob.glob('*.csv'):
    print(i)
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
order_products__prior = pd.read_csv('order_products__prior.csv')
order_products__train = pd.read_csv('order_products__train.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')
print('__Dimensions of our Data Sets__')
print('aisles'.ljust(30),aisles.shape)
print('departments'.ljust(30),departments.shape)
print('order_products__prior'.ljust(30),order_products__prior.shape)
print('order_products__train'.ljust(30),order_products__train.shape)
print('orders'.ljust(30),orders.shape)
print('products'.ljust(30),products.shape)
print('aisle Overview:')
aisles.head()
def get_aisle(id):
    return aisles[aisles['aisle_id'] == id].iloc[0,1]
print('departments Overview:')
departments.head()
def get_department(id):
    return departments[departments['department_id'] == id].iloc[0,1]
print('products overview')
products.head()
products['aisle'] = np.vectorize(get_aisle)(products['aisle_id'])
products['department'] = np.vectorize(get_department)(products['department_id'])
print('Products overview after adding aisle and department columns')
products.head()
del (aisles, departments)
print('orders Overview')
orders.head()
which_hour = orders.groupby(['order_hour_of_day'])['order_id'].count().reset_index()
plt.figure(figsize = (12,8))
sns.barplot(which_hour.order_hour_of_day, which_hour.order_id)
plt.xlabel('Hour of the Day', fontsize = 12)
plt.ylabel('Orders Count', fontsize = 12)
plt.title('Which hour more orders are placed', fontsize = 12)
plt.show()
del which_hour
which_day = orders.groupby(['order_dow'])['order_id'].count().reset_index()
plt.figure(figsize = (12,8))
sns.barplot(which_day.order_dow, which_day.order_id)
plt.xlabel('Day of the Week', fontsize = 12)
plt.ylabel('Orders Count', fontsize = 12)
plt.title('Which Day more orders are placed', fontsize = 12)
plt.show()
del which_day
orders['eval_set'].value_counts()
counts = pd.DataFrame(orders.eval_set.value_counts())

plt.figure(figsize = (12,8))
sns.barplot(counts.index, counts.eval_set)
plt.xlabel("type", fontsize = 12)
plt.ylabel("Number of Orders", fontsize = 12)
plt.title("counts of prior,train,test", fontsize = 12)
plt.show()
del counts
orders.groupby(['eval_set'])['user_id'].nunique()
user_counts = pd.DataFrame(orders.groupby(['eval_set'])['user_id'].nunique())

plt.figure(figsize = (12,8))
sns.barplot(user_counts.index,user_counts.user_id)
plt.xlabel("type", fontsize = 12)
plt.ylabel("Number of Users", fontsize = 12)
plt.title("counts of prior,train,test", fontsize = 12)
plt.show()
del user_counts
print('order_products__prior Overview:')
order_products__prior.head()
print('order_products__prior Overview:')
order_products__prior.head()
orders.isnull().sum()
order_products__prior.isnull().sum()
order_products__train.isnull().sum()
products.isnull().sum()
order_products_train = order_products__train.merge(orders[['order_id','user_id']], on = 'order_id', how = 'left')
order_products_train.head()
orders_products = orders.merge(order_products__prior, on = 'order_id', how = 'inner')
del order_products__prior
prd = orders_products.sort_values(['user_id', 'order_number', 'product_id'])
prd['product_time'] = prd.groupby(['user_id','product_id'])['order_number'].cumcount(ascending=True)
prd.head()
def uni(a):
    return a.nunique()
def zer(a):
    return sum(a==0)
def one(a):
    return sum(a==1)
prd = prd.groupby('product_id').agg({'order_id':uni, 'reordered':sum, 'product_time':[zer,one]}).reset_index()
prd.columns = [' '.join(col).strip() for col in prd.columns.values]
prd.rename(columns = {'order_id uni':'prod_orders',
                      'product_time zer':'prod_first_orders',
                      'product_time one':'prod_second_orders',
                      'reordered sum':'prod_reorders'},inplace = True)
prd['prod_reorder_probability'] = prd.prod_second_orders / prd.prod_first_orders
prd['prod_reorder_times'] = 1 + prd.prod_reorders / prd.prod_first_orders
prd['prod_reorder_ratio'] = prd.prod_reorders / prd.prod_orders
prd.drop(prd[['prod_reorders','prod_first_orders','prod_second_orders']], axis=1, inplace=True)
prd.head()
users = orders[orders['eval_set'] == 'prior']
def mean(a):
    return a.mean()

users = users.groupby(['user_id']).agg({'order_number':max,'days_since_prior_order':[sum,mean]}).reset_index()

users.columns = [' '.join(col).strip() for col in users.columns.values]
users.rename(columns = {'order_number max':'user_orders',
                        'days_since_prior_order sum':'user_period',
                        'days_since_prior_order mean':'user_mean_days_since_prior'},inplace = True)
users.head()
def count(a):
    return a.count()
def unique(a):
    return a.nunique()
def equal(a):
    return sum(a == 1)
def grater(a):
    return sum(a > 1)

us = orders_products.groupby(['user_id']).agg({'order_id':count,'reordered':equal,'order_number':grater,'product_id':unique}).reset_index()

us['user_reorder_ratio'] = us.reordered/us.order_number
us.drop(us[['reordered','order_number']], axis = 1, inplace = True)
us.rename(columns = {'order_id':'user_total_products','product_id':'user_distinct_products'}, inplace = True)
us.head()
users = users.merge(us, on = 'user_id', how = 'inner')
users['user_average_basket'] = users.user_total_products / users.user_orders
users.head()
us = orders[orders['eval_set'] != 'prior']
us = us[['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
us.rename({'days_since_prior_order':'time_since_last_order'}, inplace = True)
users = users.merge(us, on = 'user_id', how = 'inner')
del us
def count(a):
    return a.count()
def first(a):
    return a.min()
def last(a):
    return a.max()
data = orders_products.groupby(['user_id','product_id']).agg({'order_id':count,
                                                              'order_number':[first,last],
                                                              'add_to_cart_order':mean}).reset_index()
data.columns = [' '.join(col).strip() for col in data.columns.values]
data.head()
data.rename(columns = {'order_id count':'up_orders',
                       'add_to_cart_order mean':'up_average_cart_position',
                       'order_number first':'up_first_order',
                       'order_number last':'up_last_order'}, inplace = True)
data.head()
data = data.merge(prd, on = 'product_id', how = 'inner')
data = data.merge(users, on = 'user_id', how = 'inner')
data.head()
data['up_order_rate'] = data.up_orders / data.user_orders
data['up_orders_since_last_order'] = data.user_orders - data.up_last_order
data = data.merge(order_products_train[['user_id','product_id','reordered']], on = ['user_id','product_id'], how = 'left')
data.head()
del (order_products__train, prd, users)
train = data[data['eval_set'] == 'train']
train.drop(train[['eval_set','user_id','product_id','order_id']], axis = 1, inplace = True)
train = train.fillna({'reordered':0})
train.head()
test = data[data['eval_set'] == 'test']
test = test.drop(test[['eval_set','user_id','reordered']], axis = 1)
test.head()
del data
train.columns
test.columns
ind_columns = train.drop(train[['reordered']], axis = 1).columns
train_ind = train.as_matrix(ind_columns)
train_dep = train.as_matrix(['reordered'])
test_columns = test.drop(test[['order_id','product_id']], axis = 1).columns
test_ind = test.as_matrix(test_columns)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_ind,train_dep)
prediction = clf.predict(test_ind)
prediction
test['reordered'] = prediction
test.head()
result_file = test[test['reordered'] == 1]
result_file = result_file.groupby(['order_id'])['product_id'].unique().reset_index()
result_file = result_file.rename(columns = {'product_id':'products'})
result_file.head()
final_result = pd.DataFrame({'order_id':orders[orders['eval_set'] == 'test'].order_id})
final_result = final_result.merge(result_file, on = 'order_id', how = 'left')
final_result = final_result.sort_values(['order_id'])
final_result = final_result.fillna({'products':'None'})
final_result.to_csv('insta_result.csv', index = False)