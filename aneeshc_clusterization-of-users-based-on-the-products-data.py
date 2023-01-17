import pandas as pd

import numpy as np



input_folder = '../input/'



products = pd.read_csv(input_folder + 'products.csv', index_col='product_id')

orders = pd.read_csv(input_folder + 'orders.csv', usecols=['order_id','user_id','eval_set'], index_col='order_id')

item_prior = pd.read_csv(input_folder + 'order_products__prior.csv', usecols=['order_id','product_id'], index_col=['order_id','product_id'])
# basic prior products table

user_product = orders.join(item_prior, how='inner').reset_index().groupby(['user_id','product_id']).count()

user_product = user_product.reset_index().rename(columns={'order_id':'prior_order_count'})
from scipy.sparse import csr_matrix

user_product_sparse = csr_matrix((user_product['prior_order_count'], (user_product['user_id'], user_product['product_id'])), shape=(user_product['user_id'].max()+1, user_product['product_id'].max()+1), dtype=np.uint16)
from sklearn.decomposition import TruncatedSVD

decomp = TruncatedSVD(n_components=10, random_state=101)

user_reduced = decomp.fit_transform(user_product_sparse)



print(decomp.explained_variance_ratio_[:10], decomp.explained_variance_ratio_.sum())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

user_reduced_scaled = scaler.fit_transform(user_reduced)
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.05, random_state=101)

clf.fit(user_reduced_scaled)

outliers = clf.predict(user_reduced_scaled)



unique, counts = np.unique(outliers, return_counts=True)

dict(zip(unique, counts))
%matplotlib inline

import matplotlib.pyplot as plt



# red is an outlier, green is a regular observation

color_map = np.vectorize({ -1: 'r', 1: 'g'}.get)

plt.scatter(user_reduced_scaled[:,0], user_reduced_scaled[:,1], c=color_map(outliers), alpha=0.1)
from sklearn.cluster import KMeans



clusters_count = 10



kmc = KMeans(n_clusters=clusters_count, init='random', n_init=10, random_state=101)

kmc.fit(user_reduced_scaled[outliers == 1,:])

clusters = kmc.predict(user_reduced_scaled)



unique, counts = np.unique(clusters, return_counts=True)

dict(zip(unique, counts))
plt.scatter(user_reduced_scaled[:,0], user_reduced_scaled[:,1], c=clusters / (clusters_count-1), cmap='tab10', alpha=0.1)
# dataframe with overall product ranks

top_products_overall = user_product[['product_id','prior_order_count']].groupby('product_id').sum().reset_index().sort_values('prior_order_count', ascending=False)

top_products_overall['rank_overall'] = top_products_overall['prior_order_count'].rank(ascending=False)



# packing clusters we found into dataframe

usersdf = pd.DataFrame(clusters[1:], columns=['cluster'], index=np.arange(1, user_product['user_id'].max()+1))



# dataframe with product ranks across clusters

top_products = user_product.merge(usersdf, left_on='user_id', right_index=True)[['product_id','cluster','prior_order_count']].groupby(['product_id','cluster']).sum().reset_index().sort_values(['cluster','prior_order_count'], ascending=False)

top_products['rank'] = top_products[['cluster','prior_order_count']].groupby('cluster').rank(ascending=False)



# merging with overall top products

top_products = top_products.merge(top_products_overall[['product_id','rank_overall']], left_on='product_id', right_on='product_id')

# calculating differences between ranks

top_products['rank_diff'] = top_products['rank'] - top_products['rank_overall']

# leaving top products in each cluster: 2 with largest and 2 with smallest difference in ranks

top_products_asc_diff = top_products.sort_values(['cluster','rank_diff'], ascending=False).groupby('cluster').head(2).reset_index(drop=True)

top_products_desc_diff = top_products.sort_values(['cluster','rank_diff'], ascending=True).groupby('cluster').head(2).reset_index(drop=True)

top_products_diff = pd.concat([top_products_asc_diff,top_products_desc_diff], axis=0)



# printing results

top_products_diff.merge(products[['product_name']], left_on='product_id', right_index=True)[['cluster','product_name','rank','rank_overall','rank_diff']].sort_values(['cluster','rank_diff'])