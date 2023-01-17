#!pip install --upgrade pip
#!pip install apyori
import os 
import pandas as pd
import numpy as np
from apyori import apriori
from collections import Counter
from datetime import datetime
from itertools import combinations
import matplotlib.pyplot as plt
aisles = pd.read_csv('../input/instacart-market-basket-analysis/aisles.csv')
aisles.dtypes
aisles
aisles.isna().sum(axis = 0)
departments = pd.read_csv('../input/instacart-market-basket-analysis/departments.csv')
departments.dtypes
departments
departments.isna().sum(axis = 0)
products = pd.read_csv('../input/instacart-market-basket-analysis/products.csv')
products.dtypes
products
aisles[aisles['aisle_id'] == 61]
departments[departments['department_id'] == 19]
products.describe()
orders = pd.read_csv('../input/instacart-market-basket-analysis/orders.csv')
orders.dtypes
orders
orders.shape
orders.eval_set.value_counts()
orders = orders[orders.eval_set == 'prior']
orders.drop('eval_set', axis = 1, inplace = True)
orders.isna().sum(axis = 0)
orders.loc[orders.days_since_prior_order.isna()]
plt.plot(orders.order_number[:100])
plt.title('Sequence of order number')
plt.xlabel('Sequence in the dataframe')
plt.ylabel('Order Number');
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].boxplot(orders.order_dow)
ax[0].set_title('Boxplot day of week')
ax[0].set_ylabel('day of week')
ax[1].hist(orders.order_hour_of_day)
ax[1].set_title('Histogram hour of day')
ax[1].set_xlabel('hour')
ax[1].set_ylabel('count');
plt.figure(figsize = (15,5))
plt.bar(range(100), orders.days_since_prior_order[:100] + 1)
plt.title('Days since prior order')
plt.xlabel('index')
plt.ylabel('days since prior order + 1');
order_products = pd.read_csv('../input/instacart-market-basket-analysis/order_products__prior.csv')
order_products.dtypes
order_products
order_products.isna().sum(axis=0)
orders_apriori = orders.copy()
orders_user = orders.groupby('user_id')['order_number'].max() #it takes the maximum number of orders placed
orders_user.head()
products_user = orders[['order_id', 'user_id']].merge(
    order_products[['order_id', 'add_to_cart_order']].groupby('order_id').max().rename({'add_to_cart_order': 'order_size'}, axis = 1),
                                                                                        on = 'order_id')
products_user
products_user[products_user.order_id == 2]
products_user = products_user.drop('order_id', axis = 1).groupby('user_id')['order_size'].sum()
products_user
fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].hist(orders_user, bins = max(orders_user) - min(orders_user))
ax[0].set_title('Count of orders by user')
ax[0].set_xlabel('number of orders')
ax[0].set_ylabel('count')

ax[1].hist(products_user, bins = 100)
ax[1].set_title('Count of products by user')
ax[1].set_xlabel('number of products')
ax[1].set_ylabel('count');
orders_apriori.drop(['user_id', 'order_id'], axis = 1, inplace=True)
orders_apriori.head()
orders.head()
orders_by_order_number = orders.order_number.value_counts()
plt.bar(orders_by_order_number.index, orders_by_order_number)
plt.title('Number of orders by order number')
plt.xlabel('order number')
plt.ylabel('number of orders');
#Convert to categorical variables since we will work with membership rules:

def order_number_categorical(order_number):
  if order_number in range(3):
    return 'order_number_1-3'
  if order_number in range(3, 5):
    return 'order_number_4-5'
  if order_number in range(5, 10):
    return 'order_number_6-10'
  if order_number in range(10, 20):
    return 'order_number_11-20'
  if order_number in range(20, 40):
    return 'order_number_21-40'
  if order_number in range(40, 60):
    return 'order_number_41-60'
  if order_number >= 60:
    return 'order_number_60+'
orders_apriori.order_number = orders_apriori.order_number.map(order_number_categorical)
orders_apriori.head()
#Total orders per day of the week:

orders_by_dow = orders.order_dow.value_counts()
orders_by_dow
#Total products per day of the week:

products_by_dow = orders[['order_id', 'order_dow']].merge(
    order_products[['order_id', 'add_to_cart_order']].groupby('order_id').max().rename({'add_to_cart_order': 'order_size'}, axis = 1),
    on = 'order_id')
products_by_dow = products_by_dow.drop('order_id', axis=1).groupby('order_dow')['order_size'].sum()
products_by_dow
#The results in a more visual way:

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(orders_by_dow.index, orders_by_dow)
ax[0].set_title('Number of orders by day of week')
ax[0].set_xlabel('day of week')
ax[0].set_ylabel('number of orders')

ax[1].bar(products_by_dow.index, products_by_dow)
ax[1].set_title('Number of products by day of week')
ax[1].set_xlabel('day of week')
ax[1].set_ylabel('number of products');
def dow_categorical(dow):
    if dow in [0, 1]:
        return 'weekend'
    else:
        return 'weekday'
orders_apriori.order_dow = orders_apriori.order_dow.map(dow_categorical)
orders_apriori.head()
orders_by_hour = orders.order_hour_of_day.value_counts()
orders_by_hour
products_by_hour = orders[['order_id', 'order_hour_of_day']].merge(
    order_products[['order_id', 'add_to_cart_order']].groupby('order_id').max().rename({'add_to_cart_order': 'order_size'}, axis = 1),
    on = 'order_id')

products_by_hour = products_by_hour.drop('order_id', axis = 1).groupby('order_hour_of_day')['order_size'].sum()

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(orders_by_hour.index, orders_by_hour)
ax[0].set_title('Number of orders by hour of day')
ax[0].set_xlabel('hour of day')
ax[0].set_ylabel('number of orders')

ax[1].bar(products_by_hour.index, products_by_hour)
ax[1].set_title('Number of products by hour of day')
ax[1].set_xlabel('hour of day')
ax[1].set_ylabel('number of products');
# conversion to categorical:

def hour_categorical(hour):
  if hour in range(7):
    return 'early_hours'
  if hour in range(7,10):
    return 'hour_' + str(hour)
  if hour in range(10, 17):
    return 'peak_hours'
  if hour in range(17, 24):
    return 'hour_' + str(hour)
orders_apriori.order_hour_of_day = orders_apriori.order_hour_of_day.map(hour_categorical)
orders_apriori.head()
plt.hist(orders.days_since_prior_order, bins = 30)
plt.title('Histogram of days since prior order')
plt.xlabel('days')
plt.ylabel('count of days');
# conversion to categorical:

def interval_categorical(interval):
    if np.isnan(interval):
        return 'first_order'
    elif interval in [7, 14, 21]:
        return 'interval_weekly'
    elif interval == 30:
        return 'interval_30+'
    else:
        return 'interval_others'
orders_apriori.days_since_prior_order = orders_apriori.days_since_prior_order.map(interval_categorical)
orders_apriori.head()
products_id_to_name = {k: v for k, v in zip(products.product_id, products.product_name)}
print(products_id_to_name)
#create a new data frame:

order_products_names = order_products.copy()
order_products_names['product_name'] = order_products_names.product_id.map(lambda x: products_id_to_name[x])
order_products_names
#count how many times the product was purchased for the first time and how many times a product was repurchased:

reorder_proportion = pd.crosstab(order_products_names.product_name, order_products_names.reordered)
reorder_proportion
reorder_proportion.sort_values(by = 0, ascending=False)
reorder_proportion.sort_values(by = 1, ascending=False)
reorder_proportion['total'] = reorder_proportion.sum(axis = 1)
reorder_proportion['0.perc'] = reorder_proportion[0] / reorder_proportion['total']
reorder_proportion['1.perc'] = reorder_proportion[1] / reorder_proportion['total']
reorder_proportion.head()
reorder_proportion.sort_values(by = ['0.perc', 'total'], ascending = False)[['0.perc', 'total']]
reorder_proportion.sort_values(by = ['1.perc', 'total'], ascending = False)[['1.perc', 'total']]
reorder_proportion.total.sort_values(ascending=False)
products_bought = sorted(order_products.product_id.unique())
print(len(products_bought), len(products))
products_not_bought = list(products.product_id[~products.product_id.isin(products_bought)])
products_not_bought
#the name of the products not bought
[products_id_to_name[product] for product in products_not_bought]
products_not_registered = list(pd.Series(products_bought)[~pd.Series(products_bought).isin(products.product_id)])
print(len(products_not_registered), products_not_registered)
cart_size = order_products.groupby('order_id')['add_to_cart_order'].max()
cart_size = cart_size.value_counts()
plt.bar(cart_size.index, cart_size)
plt.title('Count of order size')
plt.xlabel('order size')
plt.ylabel('count');
add_to_cart = pd.crosstab(order_products_names.product_name, order_products_names.add_to_cart_order)
add_to_cart
for i in range(1,6):
    print('ORDER = ', i)
    print(add_to_cart.sort_values(by = i, ascending=False)[i][:5])
    print('\n')
orders_apriori.head()
orders_apriori.shape
trans = []
for i in range(orders_apriori.shape[0]):
    trans.append([str(orders_apriori.values[i, j]) for j in range(orders_apriori.shape[1])])
trans[:4]
start = datetime.now()
rules = apriori(trans, min_support = 0.005, min_confidence = 0.2, min_lift = 2)
results = list(rules)
print('Execution time: ', datetime.now() - start)
results[0]
#more detailed analysis of the rule:
results[0][0]
#item 0, position 1:
results[0][1]
#create a vriable r with results in position 0 and position 2:
r = results[0][2]
r
#it will return the fist rule
r[0]
#it will return the second rule
r[1]
#it return the fist rule and the confidence, after it will return the fist rule and the lift:
r[0][2], r[0][3]
A = []
B = []
support = []
confidence = []
lift = []

for result in results:
  s = result[1]
  result_rules = result[2]
  for result_rule in result_rules:
    a = list(result_rule[0])
    b = list(result_rule[1])
    c = result_rule[2]
    l = result_rule[3]
    A.append(a)
    B.append(b)
    support.append(s)
    confidence.append(c)
    lift.append(l) 

rules_df = pd.DataFrame({
    'A': A,
    'B': B,
    'support': support,
    'confidence': confidence,
    'lift': lift
})

rules_df = rules_df.sort_values(by = 'lift', ascending = False).reset_index(drop = True)
len(rules_df)
A[0], B[0], A[1], B[1]
rules_df
transactions_df = order_products[['order_id', 'product_id']][:5000]
transactions_df
n_orders = len(set(transactions_df.order_id))
n_products = len(set(transactions_df.product_id))
print(n_orders, n_products)
product_frequency = transactions_df.product_id.value_counts() / n_orders
plt.hist(product_frequency, bins = 100)
plt.title('Number of times each product frequency occurs')
plt.xlabel('product frequency')
plt.ylabel('number of times');
#a zoom:

plt.hist(product_frequency, bins = 100)
plt.title('Number of times each product frequency occurs')
plt.xlabel('product frequency')
plt.ylabel('number of times')
plt.ylim([0, 100]);
min_support = 0.01
products_apriori = product_frequency[product_frequency >= min_support]
print(products_apriori)
transactions_apriori = transactions_df[transactions_df.product_id.isin(products_apriori.index)]
transactions_apriori
order_sizes = transactions_apriori.order_id.value_counts()
order_sizes
plt.hist(order_sizes, bins = max(order_sizes) - min(order_sizes))
plt.title('Number of times each order size occurs')
plt.xlabel('order size')
plt.ylabel('number of times');
min_lenght = 2
orders_apriori = order_sizes[order_sizes >= min_lenght]
print(orders_apriori)
transactions_apriori = transactions_apriori[transactions_apriori.order_id.isin(orders_apriori.index)]
transactions_apriori
transactions_by_order = transactions_apriori.groupby('order_id')['product_id']
for order_id, order_list in transactions_by_order:
  print('Order_id:', order_id, '\nOrder_list: ', list(order_list))
  product_combinations = combinations(order_list, 2)
  print('Product combinations:')
  print([i for i in product_combinations])
  print('\n')
def product_combinations(transactions_df, max_length = 5):
  transactions_by_order = transactions_df.groupby('order_id')['product_id']
  max_length_reference = max_length
  for order_id, order_list in transactions_by_order:
    max_length = min(max_length_reference, len(order_list))
    order_list = sorted(order_list)
    for l in range(2, max_length + 1):
      product_combinations = combinations(order_list, l)
      for combination in product_combinations:
        yield combination
combs = product_combinations(transactions_apriori)
combs
#view all combinations of products that have been generated:

for _ in range(100):
  print(next(iter(combs)))
#how often each of these combinations appears:

combs = product_combinations(transactions_apriori)
counter = Counter(combs).items()
combinations_count = pd.Series([x[1] for x in counter], index = [x[0] for x in counter])
combinations_frequency = combinations_count / n_orders
print(combinations_frequency)
combinations_apriori = combinations_frequency[combinations_frequency >= min_support]
combinations_apriori = combinations_apriori[combinations_apriori.index.map(len) >= min_lenght]
print(combinations_apriori, len(combinations_apriori))
A = []
B = []
AB = []
for c in combinations_apriori.index:
  c_length = len(c)
  for l in range(1, c_length):
    comb = combinations(c, l)
    for a in comb:
      AB.append(c)
      b = list(c)
      for e in a:
        b.remove(e)
      b = tuple(b)
      if len(a) == 1:
        a = a[0]
      A.append(a)
      if len(b) == 1:
        b = b[0]
      B.append(b)
apriori_df = pd.DataFrame({'A': A,
                           'B': B,
                           'AB': AB})
apriori_df.head()
products_apriori
combinations_frequency
support = {**{k: v for k, v in products_apriori.items()},
           **{k: v for k, v in combinations_frequency.items()}}
support
#updating thevapriori_df with the news combinations:

apriori_df[['support_A', 'support_B', 'support_AB']] = apriori_df[['A', 'B', 'AB']].applymap(lambda x: support[x])
apriori_df
apriori_df.drop('AB', axis = 1, inplace=True)
apriori_df.head()
#generating confidence and lift:

apriori_df['confidence'] = apriori_df.support_AB / apriori_df.support_A
apriori_df['lift'] = apriori_df.confidence / apriori_df.support_B
apriori_df
min_confidence = 0.2
min_lift = 1.0
apriori_df = apriori_df[apriori_df.confidence >= min_confidence]
apriori_df = apriori_df[apriori_df.lift >= min_lift]
apriori_df = apriori_df.sort_values(by = 'lift', ascending=False).reset_index(drop = True) #ordering by the lift
apriori_df.head()
#getting the names of the products:

def convert_product_id_to_name(product_ids):
  if type(product_ids) == int:
    return products_id_to_name[product_ids]
  names = []
  for prod in product_ids:
    name = products_id_to_name[prod]
    names.append(name)
  names = tuple(names)
  return names
#applying the names in the data frame:

apriori_df[['A', 'B']] = apriori_df[['A', 'B']].applymap(convert_product_id_to_name)
apriori_df
def association_rules(order_products, min_support, min_length = 2, max_length = 5, 
                      min_confidence = 0.2, min_lift = 1.0):
    
    print('Loading data...')
    transactions_df = order_products[['order_id', 'product_id']]

    print('Calculating product supports...')
    n_orders = len(set(transactions_df.order_id))
    product_frequency = transactions_df.product_id.value_counts()/n_orders
    products_apriori = product_frequency[product_frequency >= min_support]
    transactions_apriori = transactions_df[transactions_df.product_id.isin(products_apriori.index)]
    
    order_sizes = transactions_apriori.order_id.value_counts()
    orders_apriori = order_sizes[order_sizes >= min_length]
    transactions_apriori = transactions_apriori[transactions_apriori.order_id.isin(orders_apriori.index)]
    
    print('Calculating product combinations and supports...')
    
    def product_combinations(transactions_df, max_length = max_length):
        transactions_by_order = transactions_df.groupby('order_id')['product_id']
        max_length_reference = max_length
        for order_id, order_list in transactions_by_order:
            max_length = min(max_length_reference, len(order_list))
            order_list = sorted(order_list)
            for l in range(2, max_length + 1):
                product_combinations = combinations(order_list, l)
                for combination in product_combinations:
                    yield combination
   
    combs = product_combinations(transactions_apriori)
    counter = Counter(combs).items()
    combinations_count = pd.Series([x[1] for x in counter], index = [x[0] for x in counter])
    combinations_frequency = combinations_count/n_orders
    combinations_apriori = combinations_frequency[combinations_frequency >= min_support]
    combinations_apriori = combinations_apriori[combinations_apriori.index.map(len) >= min_length]
    
    print('Populating dataframe...')
    A = []
    B = []
    AB = []
    for c in combinations_apriori.index:
        c_length = len(c)
        for l in range(1, c_length):
            comb = combinations(c, l)
            for a in comb:
                AB.append(c)
                b = list(c)
                for e in a:
                    b.remove(e)
                b = tuple(b)
                if len(a) == 1:
                    a = a[0]
                A.append(a)
                if len(b) == 1:
                    b = b[0]
                B.append(b)
            
    apriori_df = pd.DataFrame({'A': A,
                               'B': B,
                               'AB': AB})
    support = {**{k: v for k, v in products_apriori.items()}, 
               **{k: v for k, v in combinations_frequency.items()}}
    apriori_df[['support_A', 'support_B', 'support_AB']] = apriori_df[['A', 'B', 'AB']].applymap(lambda x: support[x])
    apriori_df.drop('AB', axis = 1, inplace = True)
    apriori_df['confidence'] = apriori_df.support_AB/apriori_df.support_A
    apriori_df['lift'] = apriori_df.confidence / apriori_df.support_B
    apriori_df = apriori_df[apriori_df.confidence >= min_confidence]
    apriori_df = apriori_df[apriori_df.lift >= min_lift]
    apriori_df = apriori_df.sort_values(by = 'lift', ascending = False).reset_index(drop = True)
    
    def convert_product_id_to_name(product_ids):
        if type(product_ids) == int:
            return products_id_to_name[product_ids]
        names = []
        for prod in product_ids:
            name = products_id_to_name[prod]
            names.append(name)
        names = tuple(names)
        return names
    
    apriori_df[['A', 'B']] = apriori_df[['A', 'B']].applymap(convert_product_id_to_name)

    print('{} rules were generated'.format(len(apriori_df)))

    return apriori_df
start = datetime.now()
rules = association_rules(order_products, min_support = 0.01)
print('Execution time: ', datetime.now() - start)
rules
start = datetime.now()
rules = association_rules(order_products, min_support = 0.009, max_length = 4)
print('Execution time: ', datetime.now() - start)
rules
start = datetime.now()
rules = association_rules(order_products, min_support = 0.002, max_length=3)
print('Execution time: ', datetime.now() - start)
rules.head(20)
start = datetime.now()
rules = association_rules(order_products, min_support = 0.001, max_length=2)
print('Execution time: ', datetime.now() - start)
rules.head(20)
rules.tail(10)