# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline
ai_df = pd.read_csv('../input/aisles.csv')

ai_df.head()
len(ai_df.aisle.unique())
de_df = pd.read_csv('../input/departments.csv')

de_df.department.values
de_df.shape
op_df = pd.read_csv('../input/order_products__prior.csv')

op_df.head()
len(op_df.order_id.unique())
op_df.shape
#in all sell record, how much is reordered

reod = op_df['reordered'][op_df['reordered']==1].sum()

nore = op_df.shape[0] - reod

print(reod, nore)
#diff reordered 0-1, bars

plt.figure(figsize=(12,6))

sns.set(style='darkgrid')

ax = sns.countplot(x='reordered', data=op_df)

plt.show()
#how much products contain in each order, tu many ,scatter just run locale

cnt_srs = (op_df['order_id'].value_counts().reset_index())['order_id'].value_counts()

plt.figure(figsize=(12,6))

plt.scatter(cnt_srs.index, cnt_srs.values)

plt.show()
#how much each product selled, bars maybe scatter

pro_cnt = op_df['product_id'].value_counts()



plt.figure(figsize=(12,6))

plt.scatter(pro_cnt.index, pro_cnt.values)

plt.xlabel('Product')

plt.ylabel('Sells Num')

plt.show()
#how much each product be reordered

rp_cnt = op_df[op_df['reordered']==1]['product_id'].value_counts()

plt.figure(figsize=(12,6))

plt.scatter(rp_cnt.index, rp_cnt.values)

plt.xlabel('Product')

plt.ylabel('Reordered Sells Num')

plt.show()
#how many product can reach the sell num

es_cnt = (pro_cnt.reset_index())['product_id'].value_counts()



plt.figure(figsize=(12,6))

plt.scatter(es_cnt.index, es_cnt.values)

plt.xlabel('Sells Num')

plt.ylabel('Product Num')

plt.ylim([0,1200])

plt.xlim([0,400])

plt.show()
#end of the visualization of order_products__prior, the added order seems quite useful, but it hard to visualize it

#begin of order_products__train

ot_df = pd.read_csv('../input/order_products__train.csv')

ot_df.head()
# 1/30 of prior

ot_df.shape
#in all sell record, how much is reordered

reodt = ot_df['reordered'][ot_df['reordered']==1].sum()

noret = ot_df.shape[0] - reodt

print(reodt, noret)
#diff reordered 0-1, bars

plt.figure(figsize=(12,6))

sns.set(style='darkgrid')

ax = sns.countplot(x='reordered', data=ot_df)

plt.show()
#how much products contain in each order, tu many ,scatter just run locale

cnt_srst = (ot_df['order_id'].value_counts().reset_index())['order_id'].value_counts()

plt.figure(figsize=(12,6))

plt.scatter(cnt_srst.index, cnt_srst.values)

plt.show()
#how much each product selled, bars maybe scatter

pro_cntt = ot_df['product_id'].value_counts()



plt.figure(figsize=(12,6))

plt.scatter(pro_cntt.index, pro_cntt.values)

plt.xlabel('Product')

plt.ylabel('Sells Num')

plt.show()
#how much each product be reordered

rp_cntt = ot_df[ot_df['reordered']==1]['product_id'].value_counts()

plt.figure(figsize=(12,6))

plt.scatter(rp_cntt.index, rp_cntt.values)

plt.xlabel('Product')

plt.ylabel('Reordered Sells Num')

plt.show()
#how many product can reach the sell num

es_cntt = (pro_cntt.reset_index())['product_id'].value_counts()



plt.figure(figsize=(12,6))

plt.scatter(es_cntt.index, es_cntt.values)

plt.xlabel('Sells Num')

plt.ylabel('Product Num')

plt.ylim([0,1200])

plt.xlim([0,400])

plt.show()
#end of order train, seems no difference with prior just less num

#begin of orders, it seems this file is ordered by user_id and order_number asc

or_df = pd.read_csv('../input/orders.csv')

or_df.head()
#prior, tarin, test

es_cnt = or_df['eval_set'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(es_cnt.index, es_cnt.values,alpha=0.8, color=color[2])

plt.xlabel('Eval Set', fontsize=12)

plt.ylabel('Order Num', fontsize=12)

plt.show()
or_df.shape
#user prefer which hour to order

ho_cnt = or_df['order_hour_of_day'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(ho_cnt.index, ho_cnt.values,alpha=0.8, color=color[3])

plt.xlabel('Hour of Day', fontsize=12)

plt.ylabel('Order Num', fontsize=12)

plt.show()
#user prefer which hour to order

#max values is 30, and it seem 30 include all values bigger here

po_cnt = or_df[or_df.days_since_prior_order.notnull()].days_since_prior_order.value_counts()

plt.figure(figsize=(12,6))

sns.barplot(po_cnt.index, po_cnt.values,alpha=0.8, color=color[3])

plt.xlabel('Days Since Prior Order', fontsize=12)

plt.ylabel('Order Num', fontsize=12)

plt.show()
#end of orders

#beginning of products

pd_df = pd.read_csv('../input/products.csv')

pd_df.head()
print(len(pd_df['aisle_id'].unique()),len(pd_df['department_id'].unique()))
#the product num of each department

dp_cnt = pd_df['department_id'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(dp_cnt.index, dp_cnt.values,alpha=0.8, color=color[3])

plt.xlabel('Department', fontsize=12)

plt.ylabel('Product Num', fontsize=12)

plt.show()
#end of products

#beginning of submissons

sb_df = pd.read_csv('../input/sample_submission.csv')

sb_df.head()
sb_df.shape
len(sb_df.order_id.unique())