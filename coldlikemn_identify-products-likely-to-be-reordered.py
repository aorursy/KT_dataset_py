#import libraries and files

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn as sk

import pandas as pd

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

aisles = pd.read_csv('../input/aisles.csv')

departments = pd.read_csv('../input/departments.csv')

df_train= pd.read_csv('../input/order_products__train.csv')

orders=pd.read_csv('../input/orders.csv')

products=pd.read_csv('../input/products.csv')
#Add a field to calculate the sum of times an item was reordered

products['rsum']=df_train.groupby('product_id')['reordered'].sum()

#Add a field to calculate the total times the item could have been reordered

products['rtotal']=df_train.groupby('product_id')['reordered'].count()

#Add a field to calculate the probability that the item was reordered

products['prob']=products['rsum']/products['rsum'].sum()

products['prob']=products['prob'].replace('0',np.nan)

products = products.dropna(how='any',axis=0)

products.head()
#Let's examine a subset of the products with the highest probability of reordering to see if there are any key attributes

g= products[(products.prob>0.003) & (products.rtotal>15)]

g = g.sort_values('prob',ascending=False)
#Add product-specific reordering probabilities to dataframe

df_train = pd.merge(left=df_train, right=products, how='left')

df_train = df_train.drop('product_name',axis=1)

df_train = pd.merge(left=df_train,right=orders,how='left')

df_train = df_train.dropna(how='any',axis=0)

df_train.head()#Let's plot the subset of the products with the highest probabilty of reordering

fig = sns.barplot(g['product_name'],g['prob'])

plt.xticks(rotation='vertical')

axe = plt.axes()

plt.ylabel('Probability Reorder', fontsize=13)

plt.xlabel('Items', fontsize=13)

plt.show()
#Get test orders and reset index

df_test = orders.loc[orders['eval_set'] == 'test']

df_test = df_test.reset_index(drop=True)

df_test.head()#Add product-specific reordering probabilities to dataframe

df_train = pd.merge(left=df_train, right=products, how='left')

df_train = df_train.drop('product_name',axis=1)

df_train = pd.merge(left=df_train,right=orders,how='left')

df_train = df_train.dropna(how='any',axis=0)

df_train.head()
#Get test orders and reset index

df_test = orders.loc[orders['eval_set'] == 'test']

df_test = df_test.reset_index(drop=True)

df_test.head()
# Create a dictionary and create very basic lists based on reorders

dict_for_df={}

for i in range(len(df_test)):

    #Select basic number of products to choose

    f= max(int(np.random.normal(5,1)),0)

    #Select products based on their overall likelihood to be reordered

    e = np.random.choice(products['product_id'],size=f,p=products['prob'])

    #Assign order_ids and products to dict

    dict_for_df[df_test['order_id'][i]]=e