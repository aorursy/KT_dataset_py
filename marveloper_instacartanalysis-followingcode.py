# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
aisles = pd.read_csv('../input/instacart-market-basket-analysis/aisles.csv')

departments = pd.read_csv('../input/instacart-market-basket-analysis/departments.csv')

order_prd_p = pd.read_csv('../input/instacart-market-basket-analysis/order_products__prior.csv')

order_prd_t = pd.read_csv('../input/instacart-market-basket-analysis/order_products__train.csv')

orders = pd.read_csv('../input/instacart-market-basket-analysis/orders.csv')

products = pd.read_csv('../input/instacart-market-basket-analysis/products.csv')
orders
orders.info(), orders.describe()
order_prd_p
order_prd_p.info(), order_prd_p.describe()
order_prd_t
order_prd_t.info(), order_prd_t.describe()
products
aisles
departments
cnt_srs = orders.eval_set.value_counts()



plt.figure(figsize = (12,8))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize =12)

plt.xlabel('Eval set type', fontsize=12)

plt.title('count of Rows in eval set type', fontsize =15)

plt.show()
def get_unique_count(x):

    return len(np.unique(x))



cnt_srs = orders.groupby('eval_set')['user_id'].aggregate(get_unique_count)

cnt_srs
cnt_srs = orders.groupby('user_id')['order_number'].aggregate(np.max).reset_index()

cnt_srs = cnt_srs.order_number.value_counts()



plt.figure(figsize = (12,8))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha = 0.8)

plt.ylabel('Number of Occurrences', fontsize =12)

plt.xlabel('Maximum order number', fontsize=12)

plt.title('count of Rows in order number', fontsize =15)

plt.xticks(rotation = 'vertical')

plt.show()
plt.figure(figsize = (12,8))

sns.countplot(x = 'order_dow', data = orders)

plt.ylabel('Count', fontsize =12)

plt.xlabel('Day of week', fontsize=12)

plt.title('Frequency of order by week day', fontsize =15)

plt.show()
plt.figure(figsize = (12,8))

sns.countplot(x = 'order_hour_of_day', data = orders)

plt.ylabel('Count', fontsize =12)

plt.xlabel('Hour of day', fontsize=12)

plt.title('Frequency of order by hour of day', fontsize =15)

plt.show()
grouped = orders.groupby(['order_dow', 'order_hour_of_day'])['order_number'].aggregate('count').reset_index()

grouped = grouped.pivot('order_dow','order_hour_of_day', 'order_number' )



plt.figure(figsize = (12,8))

sns.heatmap(grouped)

plt.ylabel('oder_dow', fontsize =12)

plt.xlabel('Hour of day', fontsize=12)

plt.title('Frequency of Day of week AND Hour of day', fontsize =15)

plt.show()
plt.figure(figsize = (12,8))

sns.countplot(x = 'days_since_prior_order', data = orders)

plt.ylabel('Count', fontsize =12)

plt.xlabel('Days since prior order', fontsize=12)

plt.title('Frequency distribution by days since prior order', fontsize =15)

plt.show()
# percentage of re-orders in prior set



order_prd_p.reordered.sum() / order_prd_p.shape[0]
# percentage of re-orders in prior set



order_prd_t.reordered.sum() / order_prd_t.shape[0]
## It takes too much RAM

# total = pd.merge(orders, order_prd_p, on = 'order_id', how='left')

# total = pd.merge(total, order_prd_t, on = 'order_id', how='left')



# total.fillna(0)  # there are NAN values where prior table has no values of train table. And vise versa.
order_prd_p = pd.merge(order_prd_p, products, on = 'product_id', how='left')

order_prd_p = pd.merge(order_prd_p, aisles, on = 'aisle_id', how='left')

order_prd_p = pd.merge(order_prd_p, departments, on = 'department_id', how='left')

order_prd_p = pd.merge(order_prd_p, orders, on = 'order_id', how='left')



order_prd_p
order_prd_t = pd.merge(order_prd_t, products, on = 'product_id', how='left')

order_prd_t = pd.merge(order_prd_t, aisles, on = 'aisle_id', how='left')

order_prd_t = pd.merge(order_prd_t, departments, on = 'department_id', how='left')

order_prd_t = pd.merge(order_prd_t, orders, on = 'order_id', how='left')



order_prd_t
print(order_prd_p.eval_set.unique(), order_prd_p.reordered.unique())
print(order_prd_t.eval_set.unique(), order_prd_t.reordered.unique())
grouped = order_prd_t.groupby('order_id')['add_to_cart_order'].aggregate('max').reset_index()

cnt_srs = grouped.add_to_cart_order.value_counts()



plt.figure(figsize = (12,8))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha = 0.8)

plt.ylabel('Number of Occurrences', fontsize =12)

plt.xlabel('Number of products in the given order', fontsize=12)

plt.title('Products in the given order', fontsize =15)

plt.show()
cnt_srs = order_prd_p['product_name'].value_counts().reset_index().head(20)

cnt_srs.columns = ['product_name', 'frequency_count']

cnt_srs
cnt_srs = order_prd_p['aisle'].value_counts().head(20)

plt.figure(figsize = (12,8))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

plt.ylabel('Number od Occurences', fontsize= 12)

plt.xlabel('Aisle', fontsize=12)

plt.xticks(rotation = 'vertical')

plt.show()
plt.figure(figsize = (10,10))

temp_series = order_prd_p['department'].value_counts()

labels = (np.array(temp_series.index))

sizes = (np.array((temp_series / temp_series.sum())* 100))

plt.pie(sizes, labels = labels, autopct = '%1.1f%%', startangle=200)

plt.show()
grouped = order_prd_p.groupby(['department'])['reordered'].aggregate('mean').reset_index()



plt.figure(figsize = (12,8))

sns.pointplot(grouped['department'].values, grouped['reordered'].values, alpha=0.8)

plt.ylabel('Reorder ratio', fontsize= 12)

plt.xlabel('Department', fontsize=12)

plt.title('Department wise reorder ratio', fontsize = 15)

plt.xticks(rotation = 'vertical')

plt.show()
grouped = order_prd_p.groupby(['department_id', 'aisle'])['reordered'].aggregate('mean').reset_index()



fig, ax = plt.subplots(figsize = (12,20))

ax.scatter(grouped.reordered.values, grouped.department_id.values)

for i, txt in enumerate(grouped.aisle.values):

    ax.annotate(txt, (grouped.reordered.values[i], grouped.department_id.values[i]), rotation = 45,

                     ha = 'center', va = 'center')

plt.ylabel('Department Id', fontsize= 12)

plt.xlabel('Reordered Ratio', fontsize=12)

plt.title('Reorder ratio of different aisles', fontsize = 15)

plt.show()
order_prd_p['add_to_cart_order_mod'] = order_prd_p['add_to_cart_order'].copy()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

AddToCart = order_prd_p['add_to_cart_order_mod']



# order_prd_p['add_to_cart_order_mod'].ix[order_prd_p['add_to_cart_order_mod']>70] = 70 ## 'ix' method doesn't work

AddToCart[AddToCart>70] = 70

grouped = order_prd_p.groupby(['add_to_cart_order_mod'])['reordered'].aggregate('mean').reset_index()



plt.figure(figsize = (12,8))

sns.pointplot(grouped['add_to_cart_order_mod'].values, grouped['reordered'].values, alpha=0.8)

plt.ylabel('Reorder ratio', fontsize= 12)

plt.xlabel('Add to cart order', fontsize=12)

plt.title('Add to cart order & Reorder ratio', fontsize = 15)

plt.xticks(rotation = 'vertical')

plt.show()
grouped = order_prd_t.groupby(['order_dow'])['reordered'].aggregate('mean').reset_index()



plt.figure(figsize = (12,8))

sns.barplot(grouped['order_dow'].values, grouped['reordered'].values, alpha=0.8)

plt.ylabel('Reorder Ratio', fontsize=12)

plt.xlabel('Day of Week', fontsize=12)

plt.title('Reorder ratio across day of week', fontsize=15)

plt.xticks(rotation='vertical')

plt.ylim(0.5, 0.7) # fix y axis height

plt.show()
grouped = order_prd_t.groupby(['order_hour_of_day'])['reordered'].aggregate('mean').reset_index()



plt.figure(figsize=(12,8))

sns.barplot(grouped['order_hour_of_day'].values, grouped['reordered'].values, alpha=0.8)

plt.ylabel('Reorder Ratio', fontsize=12)

plt.xlabel('hour of Day', fontsize=12)

plt.title('Reorder ratio across hour of day', fontsize=15)

plt.xticks(rotation='vertical')

plt.ylim(0.5, 0.7)

plt.show()
len(products.product_id.unique())
# order_prd_p.product_name.value_counts()[0:10]
# order_prd_p.aisle.value_counts()[0:10]
clst_prd = pd.crosstab(order_prd_p['user_id'], order_prd_p['aisle'])

clst_prd
from sklearn.decomposition import PCA



pca = PCA(n_components = 6)

pca.fit(clst_prd)

pca_samples = pca.transform(clst_prd)
ps = pd.DataFrame(pca_samples)

ps.head()
ps.describe()
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

tocluster = pd.DataFrame(ps[[4,1]])  # pick 2 pca columns

print(tocluster.shape)

print(tocluster.head())



fig = plt.figure(figsize=(8,8))

plt.plot(tocluster[4], tocluster[1], 'o', markersize=2, color='blue', alpha=0.5, label='class1')



plt.xlabel('x_values')

plt.ylabel('y_values')

plt.legend()

plt.show()
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



clusterer = KMeans(n_clusters = 4, random_state = 42).fit(tocluster)

centers = clusterer.cluster_centers_

c_preds = clusterer.predict(tocluster)



print(centers)
print(c_preds[0:100])
import matplotlib



fig = plt.figure(figsize = (8,8))

colors = ['orange', 'blue', 'purple', 'green']

colored = [colors[k] for k in c_preds]



print(colored[0:10])





plt.scatter(tocluster[4], tocluster[1], color = colored)



for ci,c in enumerate(centers):

    plt.plot(c[0], c[1], 'o', markersize = 8, color = 'red', alpha=0.9, label=''+str(ci))



plt.xlabel('x_values')

plt.ylabel('y_values')

plt.legend()

plt.show()
clst_prd_mod = clst_prd.copy()

clst_prd_mod['cluster'] =  c_preds



clst_prd_mod.head(10)
print(clst_prd_mod.shape)



f,arr = plt.subplots(2,2, sharex = True, figsize=(15,15))



c1_count = len(clst_prd_mod[clst_prd_mod['cluster']==0])



c0 = clst_prd_mod[clst_prd_mod['cluster']==0].drop('cluster', axis=1).mean()

arr[0,0].bar(range(len(clst_prd_mod.drop('cluster', axis=1).columns)), c0)

c1 = clst_prd_mod[clst_prd_mod['cluster']==1].drop('cluster', axis=1).mean()

arr[0,1].bar(range(len(clst_prd_mod.drop('cluster', axis=1).columns)), c1)

c2 = clst_prd_mod[clst_prd_mod['cluster']==2].drop('cluster', axis=1).mean()

arr[1,0].bar(range(len(clst_prd_mod.drop('cluster', axis=1).columns)), c2)

c3 = clst_prd_mod[clst_prd_mod['cluster']==3].drop('cluster', axis=1).mean()

arr[1,1].bar(range(len(clst_prd_mod.drop('cluster', axis=1).columns)), c3)



plt.show()
c0.sort_values(ascending=False)[0:10]
c1.sort_values(ascending=False)[0:10]
c2.sort_values(ascending=False)[0:10]
c3.sort_values(ascending=False)[0:10]
print(len(orders), len(order_prd_p),len(order_prd_t))
order_prd_p.columns
X = total[['order_id', 'user_id', 'order_number', 'order_dow',

       'order_hour_of_day', 'product_id_x',

       'add_to_cart_order_x', 'reordered_x']]

Y = total['product_id_y']
x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size = 0.3)
import tensorflow as tf

from tensorflow.keras import layers

import numpy as np
model = tf.keras.Sequential()



model.add(layers.Input(shape=8))

model.add(layers.Dense(4, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='mean_squared_error',

              optimizer = 'SGD',

              metrics = ['accuracy'])



model.fit(x_train, y_train, epochs=10, verbose=1)