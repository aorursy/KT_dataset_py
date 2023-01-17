# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import glob#Tried using it, very inconsistent tho.



from IPython.display import display

#from IPython.core.pylabtools import figsize

aisles=pd.read_csv('../input/aisles.csv')

departments=pd.read_csv('../input/departments.csv')

orderings=pd.read_csv('../input/orders.csv')

order_prior=pd.read_csv('../input/order_products__prior.csv')

order_train=pd.read_csv('../input/order_products__train.csv')

sample_submission=pd.read_csv('../input/sample_submission.csv')

products=pd.read_csv('../input/products.csv')



arr = aisles,departments,orderings,order_prior,order_train,products,sample_submission
'''

for dfs in arr:

    display(dfs.head())

    display(dfs.info())

    print('Number of null values in each dataframe')

    display(pd.isnull(dfs).sum())

'''
#Downcast the dfs to save memory and decrease runtime



#Reindex the relevant dfs

products=products.apply(lambda x: pd.to_numeric(x,errors = 'ignore',downcast = 'integer'))

order_prior = order_prior.apply(lambda x: pd.to_numeric(x,errors = 'ignore',downcast = 'integer'))

order_train = order_train.apply(lambda x: pd.to_numeric(x,errors = 'ignore',downcast = 'integer'))



#For the orders, replace NaN with -1 so we can downcast

orderings['days_since_prior_order'] = orderings['days_since_prior_order'].fillna(-1)

orderings = orderings.apply(lambda x: pd.to_numeric(x,errors = 'ignore',downcast = 'integer'))
#Reindex relevant dfs

aisles=aisles.set_index('aisle_id')

products=products.set_index('product_id')

departments=departments.set_index('department_id')
'''

#Repointing and printing everything gives:

arr = aisles,departments,orderings,order_prior,order_train,products,sample_submission

for dfs in arr:

    display(dfs.head())

    display(dfs.info())

'''
#Splitting of orderings df

priorOrderings = orderings[orderings['eval_set'] == 'prior'].drop('eval_set',1)

trainOrderings = orderings[orderings['eval_set'] == 'train'].drop('eval_set',1)

testOrderings = orderings[orderings['eval_set'] == 'test'].drop('eval_set',1)

arrOrderings=priorOrderings,trainOrderings,testOrderings

for dfs in arrOrderings:

#    display(dfs.head())

    pass
#Combining the products,aisles and departments

products_df=pd.merge(products,aisles,left_on='aisle_id',right_index=True)

products_df=pd.merge(products_df,departments,left_on='department_id',right_index=True)

products_df=products_df.sort_index()

products_df=products_df.rename(columns={'aisle':'aisle_name','department':'department_name'})

#products_df.head()
print('There are {} unique products'.format(len(products)))

print('There are {} unique departments'.format(len(departments)))

print('There are {} unique aisles'.format(len(aisles)))
sns.countplot(x = 'reordered',data=order_prior)

plt.title('Amount of reordered products')

plt.show()
order_prior_df=pd.merge(order_prior,products_df,left_on='product_id',right_index=True).sort_index()

order_reordered_df=order_prior_df[order_prior_df['reordered']==1]

order_notreordered_df=order_prior_df[order_prior_df['reordered']==0]

display(order_prior_df.head())
order_size=order_prior['order_id'].value_counts().to_frame()

order_size=order_size.rename(columns={'order_id':'order_size'})

priorOrderings=pd.merge(priorOrderings,order_size,left_on='order_id',right_index=True)

priorOrderings.head()
sns.distplot(priorOrderings.groupby('user_id')['order_size'].mean().values)

plt.xlabel('Order size')

plt.title('Average order size across users')

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(14,8))



grp=order_notreordered_df['department_name'].value_counts()

ax1.pie(grp,labels=grp.index.values,autopct='%1.1f%%',shadow = True)

ax1.set(title='Not reordered')



grp=order_reordered_df['department_name'].value_counts()

ax2.pie(grp,labels=grp.index.values,autopct='%1.1f%%',shadow = True)

ax2.set(title='Reordered')



plt.suptitle('Proportion of orders wrt. product departments')

fig.tight_layout()

plt.show()
#For the orderings df, the time usage of Instacart is given. What do people order wrt. time?

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(14,6))



grp=priorOrderings[priorOrderings['days_since_prior_order']==-1].groupby(['order_dow','order_hour_of_day'])['order_size'].mean()

sns.heatmap(grp.unstack(),cmap='cubehelix',ax=ax1)

ax1.set(title='First time users',xlabel='Hour',ylabel='DOW')



grp=priorOrderings[priorOrderings['days_since_prior_order']!=-1].groupby(['order_dow','order_hour_of_day'])['order_size'].mean()

sns.heatmap(grp.unstack(),cmap='cubehelix',ax=ax2)

ax2.set(title='Repeated users',xlabel='Hour',ylabel='DOW')



plt.suptitle('Average order size wrt. time')

plt.show()
#For the orderings df, the time usage of Instacart is given. What do people order wrt. time?

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(14,6))



grp=priorOrderings[priorOrderings['days_since_prior_order']==-1].groupby(['order_dow','order_hour_of_day'])['order_id'].count()

sns.heatmap(grp.unstack(),cmap='cubehelix',ax=ax1)

ax1.set(title='First time users',xlabel='Hour',ylabel='DOW')



grp=priorOrderings[priorOrderings['days_since_prior_order']!=-1].groupby(['order_dow','order_hour_of_day'])['order_id'].count()

sns.heatmap(grp.unstack(),cmap='cubehelix',ax=ax2)

ax2.set(title='Repeated users',xlabel='Hour',ylabel='DOW')



plt.suptitle('Total number of users that made orders wrt. time')

plt.show()
#For reorders, what is the average number of days between reordering?

grp=priorOrderings[priorOrderings['days_since_prior_order']!=-1].groupby(['order_dow','order_hour_of_day'])['days_since_prior_order'].mean()

orderings_between=grp.unstack()

sns.heatmap(orderings_between,cmap='cubehelix')

plt.xlabel('Hour')

plt.ylabel('DOW')

plt.title('Average number of days between orders')

plt.show()