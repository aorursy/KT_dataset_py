# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette('viridis')

# Any results you write to the current directory are saved as output.
# you can load one by one depends on your demand, or load them one time. 
prior = pd.read_csv("../input/order_products__prior.csv")
train = pd.read_csv("../input/order_products__train.csv")
orders = pd.read_csv("../input/orders.csv")
products =  pd.read_csv("../input/products.csv")
aisles = pd.read_csv("../input/aisles.csv")
departments = pd.read_csv("../input/departments.csv")
print('the shape of prior:', prior.shape)
print('cols of prior:', prior.columns)
print('cols of train:', train.columns)
print('the shape of orders:', orders.shape)
print('cols of orders:' ,orders.columns)
print('cols of products:', products.columns)

orders.eval_set.value_counts().plot(kind ='bar',color = color,figsize = (8,6),fontsize =12 )
# see how many unique user in each group
orders.groupby('eval_set')['user_id'].apply(lambda x: len(x.unique()))
print('cols of orders:' ,orders.columns)
fig,axes = plt.subplots(4,1,figsize = (16,24))
orders.order_dow.value_counts().plot(kind ='bar',color = 'c',ax = axes[0],title = 'Distribution of Day of Week orders')
orders.order_hour_of_day.value_counts().plot(kind ='bar',color = 'c',ax = axes[1],title = 'Distribution of Hour of Day orders')

tmp = orders.groupby(['order_dow', 'order_hour_of_day'])["order_number"].aggregate('count').reset_index()
tmp = tmp.pivot('order_dow', 'order_hour_of_day', 'order_number')
ax = axes[2]
sns.heatmap(tmp,ax = axes[2])

orders.days_since_prior_order.value_counts().plot(kind ='bar',color = 'c',ax = axes[3],title = 'Distribution of days_since_prior_order')

print('The re-ordered percentage in train dataset is: ', round(train.reordered.sum()/len(train) *100,2))
print('The re-ordered percentage in prior dataset is: ', round(prior.reordered.sum()/len(prior) *100,2))
tmp = train.groupby('order_id')['reordered'].aggregate('sum').reset_index()
tmp['reordered'].loc[tmp['reordered'] >= 1] =1
print('the percentage of non-reorders in train is ',tmp['reordered'].value_counts()/ len(tmp)) 
      
tmp = prior.groupby('order_id')['reordered'].aggregate('sum').reset_index()
tmp['reordered'].loc[tmp['reordered'] >= 1] =1
print('the percentage of non-reorders in prior is ',tmp['reordered'].value_counts()/ len(tmp)) 

tmp = train.groupby('order_id')['add_to_cart_order'].aggregate('max').reset_index()
tmp['add_to_cart_order'].value_counts()[:50].plot(kind = 'bar',legend = 'train',color = 'b',figsize =(16,6))

tmp1 = prior.groupby('order_id')['add_to_cart_order'].aggregate('max').reset_index()
tmp1['add_to_cart_order'].value_counts()[:50].plot(kind = 'bar',legend = 'prior',color = 'orange',figsize =(16,6))
cols =[products,departments, aisles]
for c in cols:
    print(c.columns)

# Build a complete products dataset
products = pd.merge(products,departments,on ='department_id', how = 'left')
products = pd.merge(products,aisles,on ='aisle_id', how = 'left')
products.head(2)
# Merge products dataset to prior dataset
#prior.columns
prior = pd.merge(prior,products,on = 'product_id',how ='left')
prior.head(2)
prior.product_name.value_counts()[:10]
prior.department.value_counts()[:10]
prior.aisle.value_counts()[:10]
tmp =prior.groupby('department')['reordered'].aggregate('mean').reset_index()
print('The top 10 goods easy to be reordered:')
display(tmp.sort_values(by ='reordered',ascending = False).head(10))
print('The top 10 goods hard to be reordered:')
display(tmp.sort_values(by ='reordered',ascending = False).tail(10))
print(len(departments),len(aisles))
# let's just pick aisles as viarable
# merge all together, because then we will do some observation between user and their order
all = pd.merge(orders,prior, on=['order_id','order_id'])
user_ai = pd.crosstab(all['user_id'], all['aisle'])
user_ai.head()
from sklearn.decomposition import PCA
pca = PCA(n_components= 10)
pca.fit(user_ai)
pca_samples = pca.transform(user_ai)
user_ai_s = pd.DataFrame(pca_samples)
user_ai_s.head()
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
tocluster = pd.DataFrame(user_ai_s[[5,1]])
print (tocluster.shape)
#print (tocluster.head())

fig = plt.figure(figsize=(8,8))
plt.plot(tocluster[5], tocluster[1], 'o', markersize=4, color='c', label='class1')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

clusterer = KMeans(n_clusters=5,random_state=42).fit(tocluster)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)
print(centers)
c_preds 
fig = plt.figure(figsize=(8,8))
colors = ['orange','g','r','cyan','yellow']
colored = [colors[k] for k in c_preds]
print (colored[0:5])
plt.scatter(tocluster[5],tocluster[1],  color = colored)
for ci,c in enumerate(centers):
    plt.plot(c[0], c[1], 'o', markersize=5, label=''+str(ci))

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
