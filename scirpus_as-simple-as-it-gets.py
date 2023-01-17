import numpy as np 

import pandas as pd 
nlargest = 3. #Just use the latest 3 orders per user



def GrabTestData():

    orders =  pd.read_csv('../input/orders.csv')

    orderstest = orders[orders.eval_set=='test']

    testusers = orderstest.user_id.values

    ordersprior = orders[orders.eval_set=='prior']

    orderstestprior = ordersprior[ordersprior.user_id.isin(testusers)]

    orderstestprior['grpids'] = range(orderstestprior.shape[0])

    grporderstestprior = orderstestprior.groupby(['user_id'])['grpids'].nlargest(int(nlargest)).reset_index()

    orderstestprior = orderstestprior[orderstestprior.grpids.isin(grporderstestprior.grpids)]

    prior = pd.read_csv('../input/order_products__prior.csv')

    orderstestprior.drop(['eval_set','grpids'],inplace=True,axis=1)

    orderstestprior = orderstestprior.merge(prior,on='order_id')

    x = orderstestprior.groupby(['user_id','product_id'])['reordered'].mean().reset_index()

    x.columns = ['user_id','product_id','romean']

    x = x[x.romean>=.5]  

    suborderstest = orders[orders.eval_set=='test']

    suborderstest.drop(['eval_set'],inplace=True,axis=1)

    suborderstest = suborderstest.merge(x,on=['user_id'])

    return suborderstest[['order_id','product_id']]
test = GrabTestData()
sub = pd.read_csv('../input/sample_submission.csv')
d2 = dict()

for row in test.itertuples():

    try:

        d2[row.order_id] += ' ' + str(row.product_id)

    except:

        d2[row.order_id] = str(row.product_id)



for order in sub.order_id:

    if order not in d2:

        d2[order] = 'None'

sub = pd.DataFrame.from_dict(d2, orient='index')

sub.reset_index(inplace=True)

sub.columns = ['order_id', 'products']

sub = sub.sort_values(by='order_id')

sub = sub.reset_index(drop=True)

sub.products = sub.products.astype(str)
sub.to_csv('simples.csv',index=False)
sub.head()