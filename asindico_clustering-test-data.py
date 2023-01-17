import numpy as np 

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
orders = pd.read_csv('../input/orders.csv')

orders = orders[orders['eval_set']=='test']

prior = pd.read_csv('../input/order_products__prior.csv')

train = pd.read_csv('../input/order_products__train.csv')

order_prior = pd.merge(prior,orders,on=['order_id','order_id'])

order_prior = order_prior.sort_values(by=['user_id','order_id'])
products = pd.read_csv('../input/products.csv')

aisles = pd.read_csv('../input/aisles.csv')

_mt = pd.merge(prior,products, on = ['product_id','product_id'])

_mt = pd.merge(_mt,orders,on=['order_id','order_id'])

mt = pd.merge(_mt,aisles,on=['aisle_id','aisle_id'])

cust_prod = pd.crosstab(mt['user_id'], mt['aisle'])



from sklearn.decomposition import PCA

pca = PCA(n_components=6)

pca.fit(cust_prod)

pca_samples = pca.transform(cust_prod)
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

tocluster = pd.DataFrame(ps[[4,1]])

print (tocluster.shape)

print (tocluster.head())



fig = plt.figure(figsize=(8,8))

plt.plot(tocluster[4], tocluster[1], 'o', markersize=2, color='blue', alpha=0.5, label='class1')



plt.xlabel('x_values')

plt.ylabel('y_values')

plt.legend()

plt.show()