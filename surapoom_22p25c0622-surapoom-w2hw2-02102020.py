# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
df.info()
df.describe()
print('ID is unique for all rows. ', (len(df['id']) == len(df['id'].unique())))
fig, ax = plt.subplots(1, 2, figsize=(20, 7))
sns.scatterplot(data=df, x='neighbourhood_group', y='price', ax=ax[0])
sns.scatterplot(data=df, x='neighbourhood', y='price', ax=ax[1])
fig, ax = plt.subplots(figsize=(20,10))
sns.scatterplot(data=df, x='latitude', y='longitude', ax=ax, hue='price', palette='seismic')
# Why there are some zero price?
df[df['price'] < 10]
# To predict price 
# filter out null last review
df[(df['last_review'].isna() == True)]
# drop 
df = df.drop(df[df['last_review'].isna() == True].index)
df['last_review'] = pd.to_datetime(df['last_review'])
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy

from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold

from sklearn import metrics
def plot_dendrogram(model, plot_label=None, **kwargs):
    # Create Linkage matrix and plot dendrogram
    
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1    #left node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
        
    linkage_matrix = np.column_stack([model.children_.astype(float), 
                                      model.distances_.astype(float), 
                                      counts.astype(float)])
        
    my_return = dendrogram(linkage_matrix, **kwargs)
    return my_return
df.columns
x_col = ['price', 'neighbourhood_group', 'neighbourhood', 'latitude', 'longitude', 'room_type',
       'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
       'availability_365']
# y_col = ['price']
non_num_col = ['neighbourhood_group', 'neighbourhood', 'room_type']
num_col = ['price', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
           'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
df_sample = df.sample(frac=1.0)
X = df_sample[x_col]
# y = df_sample[y_col]

# 'ward', 'average', 'complete', 'single'
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = AgglomerativeClustering(distance_threshold= 0, n_clusters=None)
# model = FeatureAgglomeration(distance_threshold= 0, n_clusters=None)

ct = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(), make_column_selector(dtype_include=object))
], remainder='passthrough')

model.fit_predict(ct.fit_transform(X).todense())
print(f'num of cluster = ', model.n_clusters_, ', min distance = ', model.distances_.min(), 
      'max distance = ', model.distances_.max())
fig, ax = plt.subplots(figsize=(20,7))
mydict = plot_dendrogram(model, ax=ax, truncate_mode='level', p=6, 
                leaf_font_size=10, leaf_rotation=90, show_leaf_counts=True)
mydict.values()
mydict.keys()
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(ct.fit_transform(X).todense())
plt.figure(figsize=(10, 7))
plt.scatter(X.index, X['price'], c=cluster.labels_, cmap='rainbow')

