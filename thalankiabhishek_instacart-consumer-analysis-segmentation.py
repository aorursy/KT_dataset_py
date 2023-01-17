import pandas as pd
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
os.chdir('../input/instacart-market-basket-analysis')

!pip install mlxtend --upgrade --no-deps -qq
!pip install squarify pywaffle -qq
import squarify
from pywaffle import Waffle
columns=[]
content=[]
for csv in tqdm(os.listdir()):
  if csv.endswith('.csv') and not csv.startswith('sample'):
    content.append([i for i in pd.read_csv(csv).columns.tolist()])
    columns.append(csv)
pd.DataFrame(sorted(content), index=sorted(columns)).T
products=pd.read_csv('products.csv').merge(pd.read_csv('aisles.csv'), on='aisle_id').merge(pd.read_csv('departments.csv'), on='department_id')
products.drop(['aisle_id', 'department_id'], 1, inplace=True)
products.head()
fig = plt.figure( 
    FigureClass = Waffle, 
    rows = 10, 
    values = round(products.department.value_counts()/100), 
    labels = products.department.value_counts().index.tolist(), 
    legend={'loc': 'center', 'bbox_to_anchor': (0.5, -0.3), 'ncol': 5, 'framealpha': 0, 'fontsize':15}, 
    dpi=100, figsize=(20, 10))
grouped_df = products.groupby('department')
fig, axes = plt.subplots(nrows=10, ncols=2, dpi=100, figsize=(40, 60))
for key, ax in zip(grouped_df.groups.keys(), axes.flatten()):
  group = grouped_df.get_group(key).aisle.value_counts(sort=True).reset_index()
  sns.barplot(y = 'index', x = 'aisle', 
                data = group, 
                ax=ax, palette='muted')
  ax.set_title(f"Aisle: {key.title()}", fontsize=15, fontstyle='oblique')
  ax.set_yticklabels([i.get_text().title() for i in ax.get_yticklabels()], fontsize=15)
  ax.set_ylabel(None)
  ax.set_xlabel(None)
  ax.set_xticklabels([])

  for i, value in group.iterrows():
    ax.text(x=20, y=i, s=value.aisle, verticalalignment='center', fontsize=15, fontstyle='oblique')

fig.tight_layout(h_pad=2.5)
fig.suptitle(t="Which Department is in this Aisle", fontsize=35, fontstyle='oblique', x=0.5, y=1.01)


department = products.groupby('department').size().reset_index(name='counts')

plt.figure(figsize=(20,10), dpi= 100)
squarify.plot(label=[i.title() for i in department.department], 
              sizes=department.counts**0.5, 
              color = [plt.cm.Spectral(i/float(department.shape[0])) for i in range(department.shape[0])])

plt.axis('off')
orders = pd.read_csv('orders.csv')
orders.head()
f, a = plt.subplots(nrows=1, ncols=2, figsize=(18, 5), dpi=100)
sns.countplot(x = "order_dow", data=orders, ax=a[0], palette='muted')
sns.countplot(x = "order_hour_of_day", data=orders, ax=a[1], palette='muted')
day_time_orders = pd.crosstab(orders.order_hour_of_day, orders.order_dow, normalize='columns')
day_time_orders.index = [pd.to_datetime(i, format='%H').time() for i in range(24)]
day_time_orders.columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
day_time_orders.head()

plt.figure(figsize=(10, 10), dpi=100)
sns.heatmap(data = day_time_orders, cmap='coolwarm', robust=True)
cust_seg = pd.merge(pd.read_csv('order_products__prior.csv'), products, on='product_id')
cust_seg = pd.merge(cust_seg, orders[orders.eval_set=='prior'], on='order_id')

cust_seg.drop(['order_id', 'product_id', 'add_to_cart_order', 
               'reordered', 'eval_set', 'order_number', 
               'order_dow', 'order_hour_of_day', 'days_since_prior_order'], 
              axis=1, inplace=True)
cust_seg.drop_duplicates(inplace=True)
cust_seg.head()
f"There are {cust_seg.department.nunique()} Departments with {cust_seg.product_name.nunique()} Products along {cust_seg.aisle.nunique()} Aisles with a customer base of {cust_seg.user_id.nunique()}"
cust_seg.product_name.value_counts().nlargest(10).plot(kind='bar')
cust_seg.department.value_counts().nlargest(10).plot(kind='bar')
cust_seg.aisle.value_counts().nlargest(10).plot(kind='bar')
cust_aisle = pd.crosstab(cust_seg.user_id, cust_seg.aisle)
cust_aisle.head()
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca_aisle = pd.DataFrame(pca.fit_transform(cust_aisle))
pca_aisle.head()
sns.pairplot(pca_aisle, diag_kind=None, height=10, aspect=1.5)
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
f, a = plt.subplots(nrows=5, ncols=3, figsize=(15, 10), dpi=100)
for ax, combo in tqdm(zip(a.flatten(), list(combinations(pca_aisle.columns.tolist(), 2)))):
  tocluster = pd.DataFrame(pca_aisle[list(combo)].values, columns=['PCA1', 'PCA2'])
  clusterer = KMeans(n_clusters=4, init='k-means++', n_jobs=-1).fit(tocluster)
  tocluster['Cluster'] = clusterer.predict(tocluster)
  sns.scatterplot(data=tocluster, x= 'PCA1', y= 'PCA2', hue= 'Cluster', palette='coolwarm', ax=ax)
  ax.set_xlabel('')
  ax.set_ylabel('')
  ax.set_xticklabels('')
  ax.set_yticklabels('')
  ax.set_title(f"PCA Feature:\n{combo[0]} Vs. {combo[-1]}")
handles, labels = ax.get_legend_handles_labels()
f.legend(handles, labels, loc='center', bbox_to_anchor = (1.1, 0.5), bbox_transform = plt.gcf().transFigure)
tocluster = pd.DataFrame(pca_aisle[[4,1]].values, columns=['PCA1', 'PCA2'])
wc = {}
for i in tqdm(range(3, 16)):
  clusterer = KMeans(n_clusters= i, init='k-means++', n_jobs=-1)
  cluster_labels = clusterer.fit_predict(tocluster)
  wc.update({i : clusterer.inertia_})

plt.figure(figsize=(15,10))
sns.lineplot(x = list(wc.keys()), y=list(wc.values()))
clusterer = KMeans(n_clusters= 5, init='k-means++', n_jobs=-1)
cluster_labels = clusterer.fit_predict(tocluster)
cust_aisle['Cluster'] = cluster_labels
cust_aisle.head()
# print(silhouette_score(tocluster.iloc[:, :-1].values, cluster_labels), clusterer.inertia_)
sns.countplot(data= cust_aisle.reset_index(), y= 'Cluster')
clust = cust_aisle.groupby('Cluster')
f, a = plt.subplots(nrows=5, sharex=False, figsize=(10, 20), dpi=100)
for k, ax in zip(clust.groups.keys(), a.flatten()):
  sns.barplot(data = clust.get_group(k).sum().nlargest(20).reset_index(name='count'), x='count', y = 'aisle', ax=ax)
plt.tight_layout()
from wordcloud import WordCloud
clust = cust_aisle.groupby('Cluster')
f, a = plt.subplots(nrows=5, sharex=False, figsize=(10, 20), dpi=100)
wcloud = WordCloud(colormap="coolwarm")
for k, ax in zip(clust.groups.keys(), a.flatten()):
  wcloud.generate_from_frequencies(clust.get_group(k).drop('Cluster', axis=1).sum())
  ax.imshow(wcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()