import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

%matplotlib inline



pd.set_option('display.max_columns', 120)
with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")



df.head()
features = [x for x in df.columns.values if x not in ['id','y','timestamp']]
def makeBinaryNaN(x):

    if x == x: # if x == x is true then is not NaN, so return 0

        return 0.

    else: # if NaN return 1

        return 1.

    

dfnan = df.applymap(makeBinaryNaN)

dfnan['id'] = df['id']

dfnan['timestamp'] = df['timestamp']

dfnan.head()
ids = dfnan['id'].unique()
df_agg_signal = pd.DataFrame()

for id in ids:

    df_signal = dfnan[dfnan['id'] == id][features].apply(lambda x: abs(x-x.shift()))

    df_signal.dropna(inplace=True)

    df_agg_signal[id] = df_signal.sum()



df_agg_signal.head()
listofIdwithMiddleNaN = []



for id in df_agg_signal.columns:

    if ([True] in (df_agg_signal[id].values > 1.)):

        listofIdwithMiddleNaN.append(theid)
len(listofIdwithMiddleNaN)
dfnan['counter'] = [1]*dfnan.shape[0]
dfnan_gb = dfnan[['id']+['counter']+features].groupby('id').agg('sum')

dfnan_gb.head()
df_rel_nan = dfnan_gb.apply(lambda x: x/x[0], axis=1)

df_rel_nan.head()
plt.figure()

sns.heatmap(df_rel_nan[features].as_matrix())
df_rel_nan_t = df_rel_nan[features].T

df_rel_nan_t.head()
corr = df_rel_nan_t.corr()

corr.head()
plt.figure(figsize=(14,14))

sns.clustermap(corr.as_matrix())
from sklearn import cluster



_, labels = cluster.affinity_propagation(corr.as_matrix())

n_labels = labels.max()

n_clusters = n_labels + 1
Ids = df_rel_nan.index



for i in range(n_labels + 1):

    print('Cluster {}: {}'.format((i), ', '.join( str(e) for e in list(Ids[labels == i]) )   ))
for i in range(n_labels + 1):

    print('Cluster {}: {}'.format((i), len(list(Ids[labels == i]))))
from sklearn.cluster import KMeans



n_clusters = 5



labels_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(corr.as_matrix())
Ids = df_rel_nan.index



for i in range(n_clusters):

    print('Cluster {}: {}'.format((i), ', '.join( str(e) for e in list(Ids[labels_kmeans == i]) )   ))
for i in range(n_clusters):

    print('Cluster {}: {}'.format((i), len(list(Ids[labels_kmeans == i]))))
for i in range(n_clusters):

    Idclus = list(Ids[labels_kmeans == i])

    dfclus = df_rel_nan[features].loc[Idclus].head()

    stripe = dfclus.sum().values.astype(float)/dfclus.sum().values.max()

    plt.figure(figsize = (10,0.5))

    sns.heatmap([list(stripe)], square=True, yticklabels = False, cbar =False)

    plt.text(7, 1.01,'Cluster {}'.format(str(i)), 

             fontsize=10, horizontalalignment='right', color='green', verticalalignment='bottom')

    plt.text(20, 1.01,'Members : {} ...'.format(', '.join( str(e) for e in list(Ids[labels_kmeans == i])[:10] )),

             fontsize=10, color='red', verticalalignment='bottom')