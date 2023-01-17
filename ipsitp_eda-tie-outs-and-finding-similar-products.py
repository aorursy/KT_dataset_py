# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

input_files = os.listdir("../input")
aisles = pd.read_csv('../input/aisles.csv')

departments = pd.read_csv('../input/departments.csv')

orders = pd.read_csv('../input/orders.csv')

order_products_prior = pd.read_csv('../input/order_products__prior.csv')

order_products_train = pd.read_csv('../input/order_products__train.csv')

products = pd.read_csv('../input/products.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

aisles.columns
aisles.head()
print('Number of Unique Values : {0}'.format(len(set(aisles['aisle']))))
departments.columns
departments.head()
print('Number of Unique Values : {0}'.format(len(set(departments['department']))))
departments['department']
orders.columns
orders.head()
set(orders['eval_set'])
sns.countplot(x='order_number', data=orders, palette="Greens_d")
sns.countplot(x='order_dow', data=orders, palette="Greens_d")
sns.countplot(x='order_hour_of_day', data=orders, palette="Greens_d")
sns.violinplot(x="order_dow", y="order_hour_of_day", data=orders, split=True, palette="Greens_d")
sns.countplot(y='days_since_prior_order', data=orders, palette="Greens_d")
order_products_prior.head()
len(set(order_products_prior['product_id']))
sns.countplot(x='reordered',data = order_products_prior,palette = "Greens_d")
products.head()
order_products_train.head()
order_products_train_product = pd.merge(order_products_train,products,how="left",left_on='product_id',right_on='product_id')

order_products_train__product__order = pd.merge(order_products_train_product,orders,how="left",left_on='order_id',right_on='order_id')

order_products_train__product__order__department = pd.merge(order_products_train__product__order,departments,how="left",left_on='department_id',right_on='department_id')

train_set = pd.merge(order_products_train__product__order__department,aisles,how="left",left_on='aisle_id',right_on='aisle_id')

train_set['product_id'] = train_set['product_id'].map(str) 
products = products[['product_id','product_name']]

products['product_id'] = products['product_id'].map(str)

products_dict = products.set_index('product_id')['product_name'].to_dict()
order_lists = train_set[['user_id','order_id','product_id']].groupby('order_id').apply(lambda x : " ".join(x['product_id']))

order_lists = pd.DataFrame(order_lists).reset_index()

order_lists.columns = ['order_id','products']

order_list = pd.merge(order_lists,orders[['order_id','user_id']],how="left",left_on='order_id',right_on='order_id')

order_list = order_list.sort_values('user_id')

user_product_list = order_list.groupby('user_id').apply(lambda x : " ".join(x['products']))

user_product_list = pd.DataFrame(user_product_list)

user_product_list=user_product_list.reset_index()

user_product_list.columns = ['user_id','products']
from nltk.tokenize import word_tokenize

corpus = [word_tokenize(s) for s in user_product_list['products']]
from gensim.models import Word2Vec

min_count = 200

size = 10

window = 10

model = Word2Vec(corpus, min_count=min_count, size=size, window=window)

products_desc = [(i,products_dict[i]) for i in list(model.wv.vocab.keys())]

products_desc[:20]
product_no = '33787'



similar_products = [products_dict[i[0]] for i in  model.most_similar(positive=[product_no])]

print("Similar Products for\n \"{0}\" :".format(products_dict[product_no]))

similar_products