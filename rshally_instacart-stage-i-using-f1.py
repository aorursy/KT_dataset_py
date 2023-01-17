import pandas as pd

import numpy as np

from collections import Counter



myfolder = '../input/'

prior = pd.read_csv(myfolder + 'order_products__prior.csv', dtype={'order_id': np.uint32,

           'product_id': np.uint16}).drop(['add_to_cart_order', 'reordered'], axis=1)

orders = pd.read_csv(myfolder + 'orders.csv', dtype={'order_hour_of_day': np.uint8,

           'order_number': np.uint8, 'order_id': np.uint32, 'user_id': np.uint32,

           'days_since_prior_order': np.float16}).drop(['order_dow','order_hour_of_day'], axis=1)

orders.set_index('order_id', drop=False, inplace=True)
#This might take a minute - adding the past products to the orders frame



orders['prod_list'] = prior.groupby('order_id').aggregate({'product_id':lambda x: list(x)})

orders=orders.fillna('')

orders['num_items'] = orders['prod_list'].apply(len).astype(np.uint8)

    
#aggregate again by creating a list of list of all products in all orders for each user



all_products = orders.groupby('user_id').aggregate({'prod_list':lambda x: list(x)})

all_products['mean_items']= orders.groupby('user_id').aggregate({'num_items':lambda x: np.mean(x)}).astype(np.uint8)

all_products['max_items']= orders.groupby('user_id').aggregate({'num_items':lambda x: np.max(x)}).astype(np.uint8)

all_products['user_id']=all_products.index

# This function flattens the list of list (of product_ids), then finds the most common elements in it

# and joins them into the required format for the test set only



def myfrequent(x):

    prodids = x.prod_list

    n=x.mean_items

    C=Counter( [elem for sublist in prodids for elem in sublist] ).most_common(n)

    return ' '.join(str(C[i][0]) for i in range(0,n))  



test=orders[['order_id','user_id']].loc[orders['eval_set']=='test']

test=test.merge(all_products,on='user_id')

test['products']=test.apply(myfrequent,axis=1)

test[['order_id','products']].to_csv('mean_submission0.csv', index=False)  

test.head(3)
train=orders[['order_id','user_id']].loc[orders['eval_set']=='train']

train_orders = pd.read_csv(myfolder + 'order_products__train.csv', dtype={'order_id': np.uint32,

           'product_id': np.uint16, 'reordered': np.int8}).drop(['add_to_cart_order'], axis=1)

train_orders = train_orders[train_orders['reordered']==1].drop('reordered',axis=1)  # predicting for reordered only

train['true'] = train_orders.groupby('order_id').aggregate({'product_id':lambda x: list(x)})

train['true']=train['true'].fillna('')

train['true_n'] = train['true'].fillna('').apply(len).astype(np.uint8)

train=train.merge(all_products,on='user_id')

train['prod_list']=train['prod_list'].map(lambda x: [elem for sublist in x for elem in sublist])

def myfrequent2(x):     # select the n most common elements from the prod_list

    prodids = x.prod_list

    n=x.mean_items

    C=Counter(prodids).most_common(n)

    return list((C[i][0]) for i in range(0,n))  



def f1_score_single(x):    #copied from LiLi

    y_true = set(x.true)

    y_pred = set(x.prediction)

    cross_size = len(y_true & y_pred)

    if cross_size == 0: return 0.

    p = 1. * cross_size / len(y_pred)

    r = 1. * cross_size / len(y_true)

    return 2 * p * r / (p + r)



train['prediction']=train.apply(myfrequent2,axis=1)

train['f1']=train.apply(f1_score_single,axis=1).astype(np.float16)

print('The F1 score on the traing set is  {0:.3f}.'.format(  train['f1'].mean()  ))

train.head(3)


def f1(y_true,y_pred):    

    y_true = set(y_true)

    y_pred = set(y_pred)

    cross_size = len(y_true & y_pred)

    if cross_size == 0: return 0.

    p = 1. * cross_size / len(y_pred)

    r = 1. * cross_size / len(y_true)

    return 2 * p * r / (p + r)



y_true=[1,2,3,4,5,6]

y_pred=[1,2,7,8,9,10]

print (' True, Pred, F1:   ',y_true,y_pred,f1(y_true, y_pred))

y_pred.extend([11,12,3])

print (' True, Pred, F1:   ',y_true,y_pred,f1(y_true, y_pred))

y_pred=[1,2,3,8,9,10]

print (' True, Pred, F1:   ',y_true,y_pred,f1(y_true, y_pred))
