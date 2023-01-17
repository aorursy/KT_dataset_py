#Import libraries



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
ord_prod_train_df = pd.read_csv("../input/order_products__train.csv")

ord_prod_prior_df = pd.read_csv("../input/order_products__prior.csv")

orders_df = pd.read_csv("../input/orders.csv")

products_df = pd.read_csv("../input/products.csv")

dept_df = pd.read_csv("../input/departments.csv")

aisles_df = pd.read_csv("../input/aisles.csv")
orders_df.head()
orders_in_1_order_id = orders_df[(orders_df.order_id <10)].sort_values(by =['order_id'])

orders_in_1_order_id  

#so each order_id is unique and
orders_df.info()
orders_df.hist(bins=12,figsize = (10,10))
cnt_eval = orders_df.eval_set.value_counts()



plt.figure(figsize=(12,8))

sns.barplot(cnt_eval.index, cnt_eval.values, alpha=0.8, color='blue')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Eval set type', fontsize=12)

plt.title('Count of rows in each dataset', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
cnt_eval
ord_prod_prior_df.head()
items_in_1_order_id = ord_prod_prior_df[(ord_prod_prior_df.order_id <3)].sort_values(by =['order_id'])

items_in_1_order_id
ord_prod_train_df.head()
ord_prod_prior_df.info()
ord_prod_train_df.info()
products_df.head()
products_df.info()
dept_df.info()
aisles_df.head()
aisles_df.info()
all_products= pd.merge(left=products_df, right= dept_df, left_on='department_id',right_on='department_id', how = 'left')

all_products= pd.merge(left=all_products, right= aisles_df, left_on='aisle_id', right_on='aisle_id', how = 'left')

all_products.head()
# in total we have 49678 types of product,20 department and 133 different aisles.

all_products.info()
#aggregate training dataset transactions

train_transactions= pd.merge(left=ord_prod_train_df, right= all_products, left_on='product_id',right_on='product_id', how = 'left')

prior_transactions= pd.merge(left=ord_prod_prior_df, right= all_products, left_on='product_id',right_on='product_id', how = 'left')
train_transactions.info()
prior_transactions.info()
orders_df.info()
df_all = pd.concat([train_transactions, prior_transactions], axis=0)



print("The order_products_all size is : ", df_all.shape)
df_all.head(20)
orders_df[(orders_df.order_id == 1)]
orders_df.shape
df_all.shape
total = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])

print (missing_data)
def get_unique_count(x):

    return len(np.unique(x))



cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)

cnt_srs
# Check the number of unique orders and unique products

orders_Unique = len(set(orders_df.order_id))

products_Unique = len(set(all_products.product_id))

print("There are %s orders for %s products" %(orders_Unique, products_Unique))
### aggregate by count of product for each order

no_products_bought = df_all.groupby('order_id')['product_id'].count()

no_products_bought.describe()
print ( "Averagely , users order ",(str(int(no_products_bought.mean()))) , " items per order")

print ("In median , users order ", (str(int(no_products_bought.median()))), " items per order")

print ("Minimum users order ",(str(int(no_products_bought.min()))), " items per order")

print ("Maximum users order ",(str(int(no_products_bought.max()))), " items per order")
# creating dataframe for number of products bought per order by order id

no_products_bought = pd.DataFrame(no_products_bought )

no_products_bought['order_id2'] = no_products_bought.index

no_products_bought = no_products_bought.rename(columns={'product_id': 'number_of_products_ordered'})

npb = no_products_bought.groupby(['number_of_products_ordered']).count()

npb = npb.rename(columns={'order_id2': 'number_of_users'})

npb['no_of_products_ordered'] =  npb.index

npb.head()
%%time

sns.set_style('whitegrid')

f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='vertical')

sns.barplot(x='no_of_products_ordered', y='number_of_users',  data= npb, color='grey')

plt.xlim(0,60)

plt.ylabel('Number of Orders', fontsize=13)

plt.xlabel('Number of products added in order', fontsize=13)

plt.show()
%%time

plt.figure(figsize=(12,8))

sns.countplot(x="order_dow", data=orders_df, color='grey')



plt.ylabel('Count', fontsize=12)

plt.xlabel('Day of week', fontsize=12)

plt.xticks( rotation='vertical')

plt.title("Frequency of order by week day", fontsize=15)



plt.show()
%%time

plt.figure(figsize=(12,8))

sns.countplot(x="order_hour_of_day", data=orders_df, color='grey')



plt.ylabel('Count', fontsize=12)

plt.xlabel('Hour', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of time of the day", fontsize=15)



plt.show()
%%time

plt.figure(figsize=(12,8))

sns.countplot(x="days_since_prior_order", data=orders_df, color='grey')



plt.ylabel('Count', fontsize=12)

plt.xlabel('Days Since Prior Order', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Days Since Prior Order", fontsize=15)



plt.show()
# validify with heatmap (1) all record (2) most active time : day 0 & 1,
grouped_df = orders_df.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()

grouped_df.head()
grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')

grouped_df.head()
plt.figure(figsize=(12,6))

sns.heatmap(grouped_df, cmap="YlGnBu")

plt.title("Frequency of Day of week Vs Hour of day")

plt.show()

active_subset_7 = orders_df[(orders_df.days_since_prior_order == 7.0)]

active_subset_30 = orders_df[(orders_df.days_since_prior_order == 30.0)]

active_subset = pd.concat([active_subset_7,active_subset_30])
grouped2_df = active_subset.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()

grouped2_df = grouped2_df.pivot('order_dow', 'order_hour_of_day', 'order_number')

grouped2_df.head()
plt.figure(figsize=(12,6))

sns.heatmap(grouped2_df, cmap="YlGnBu")

plt.title("Frequency of Day of week Vs Hour of day")

plt.show()
# Popular Departments

popular_departments = df_all['department'].value_counts().reset_index()

popular_departments.columns = ['department', 'frequency_count']

popular_departments.head(20)
df_all.head()
plt.figure(figsize=(12,8))

sns.countplot(x="department" ,data=df_all, order = df_all["department"].value_counts().index[:12])



plt.ylabel('Frequency Count', fontsize=12)

plt.xlabel('Department', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Popular Department", fontsize=15)



plt.show()
# Popular Departments

popular_aisles = df_all['aisle'].value_counts().reset_index()

popular_aisles.columns = ['aisle', 'frequency_count']

popular_aisles.head(20)
plt.figure(figsize=(12,8))

sns.countplot(x="aisle" ,data=df_all, order = df_all["aisle"].value_counts().index[:12])



plt.ylabel('Frequency Count', fontsize=12)

plt.xlabel('Aisle', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Popular Aisles", fontsize=15)



plt.show()
#popular Items

popular_items = df_all['product_name'].value_counts().reset_index()

popular_items.columns = ['product_name', 'frequency_count']

popular_items.head(20)
plt.figure(figsize=(12,8))

sns.countplot(x="aisle" ,data=df_all, order = df_all["aisle"].value_counts().index[:12])



plt.ylabel('Frequency Count', fontsize=12)

plt.xlabel('Product Name', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Popular Product", fontsize=15)



plt.show()
grouped = df_all.groupby(["department", "aisle"])["order_id"].aggregate({'Total_orders': 'count'}).reset_index()

grouped.sort_values(by='Total_orders', ascending=False, inplace=True)

fig, axes = plt.subplots(7,3, figsize=(20,45), gridspec_kw =  dict(hspace=1.4))

for (aisle, group), ax in zip(grouped.groupby(["department"]), axes.flatten()):

    g = sns.barplot(group.aisle, group.Total_orders , ax=ax)

    ax.set(xlabel = "Aisles", ylabel=" Number of Orders")

    g.set_xticklabels(labels = group.aisle,rotation=90, fontsize=12)

    ax.set_title(aisle, fontsize=15)
len(df_all['aisle'].unique())
df_all['aisle'].value_counts()[0:10]
a_orders= pd.merge(left=df_all, right= orders_df, left_on='order_id',right_on='order_id', how = 'left')
df = a_orders.drop(['eval_set','order_number','order_dow','order_hour_of_day','days_since_prior_order'], 1)
df.head()
all_purchases = pd.crosstab(df['user_id'], df['aisle'])

all_purchases.head()
all_purchases.shape
from sklearn.decomposition import PCA

pca = PCA(n_components=8)

pca.fit(all_purchases)

pca_samples = pca.transform(all_purchases)
ps = pd.DataFrame(pca_samples)

ps.head()
ps.shape
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

tocluster = pd.DataFrame(ps[[4,1]])  ## PC1  and #PC4 samples

print (tocluster.shape)

print (tocluster.head())



fig = plt.figure(figsize=(8,8))

plt.plot(tocluster[4], tocluster[1], 'o', markersize=2, color='blue', alpha=0.5, label='class1')



plt.xlabel('x_values')

plt.ylabel('y_values')

plt.legend()

plt.show()
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



clusterer = KMeans(n_clusters=4,random_state=42).fit(tocluster)

centers = clusterer.cluster_centers_

c_preds = clusterer.predict(tocluster)

print(centers)
print (c_preds[0:100])
import matplotlib

fig = plt.figure(figsize=(8,8))

colors = ['orange','blue','purple','green']

colored = [colors[k] for k in c_preds]

print (colored[0:10])

plt.scatter(tocluster[4],tocluster[1],  color = colored)

for ci,c in enumerate(centers):

    plt.plot(c[0], c[1], 'o', markersize=8, color='red', alpha=0.9, label=''+str(ci))



plt.xlabel('x_values')

plt.ylabel('y_values')

plt.legend()

plt.show()
all_purchases_cluster = all_purchases.copy()

all_purchases_cluster['cluster'] = c_preds



all_purchases_cluster.head(10)
print (all_purchases_cluster.shape)

f,arr = plt.subplots(2,2,sharex=True,figsize=(15,15))



c1_count = len(all_purchases_cluster[all_purchases_cluster['cluster']==0])



c0 = all_purchases_cluster[all_purchases_cluster['cluster']==0].drop('cluster',axis=1).mean()

arr[0,0].bar(range(len(all_purchases_cluster.drop('cluster',axis=1).columns)),c0)



c1 = all_purchases_cluster[all_purchases_cluster['cluster']==1].drop('cluster',axis=1).mean()

arr[0,1].bar(range(len(all_purchases_cluster.drop('cluster',axis=1).columns)),c1)



c2 = all_purchases_cluster[all_purchases_cluster['cluster']==2].drop('cluster',axis=1).mean()

arr[1,0].bar(range(len(all_purchases_cluster.drop('cluster',axis=1).columns)),c2)



c3 = all_purchases_cluster[all_purchases_cluster['cluster']==3].drop('cluster',axis=1).mean()

arr[1,1].bar(range(len(all_purchases_cluster.drop('cluster',axis=1).columns)),c3)

plt.show()
c0.sort_values(ascending=False)[0:10]
c1.sort_values(ascending=False)[0:10]
c2.sort_values(ascending=False)[0:10]
c3.sort_values(ascending=False)[0:10]