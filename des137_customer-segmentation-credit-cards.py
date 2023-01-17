import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
df_original = pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv', index_col='CUST_ID')

df = df_original.copy()
df.shape
df.columns
df.sample(10)
df.info()
df.isnull().sum()
# Filling out all the null values using median 

# More appropriate strategies might be required depending on the context

df.fillna(df.median(), inplace=True)
for col in df.columns:

    print('{:33} : {:6} : {:}'.format(col, df[col].nunique(), df[col].dtype))
(1e2*df['TENURE'].value_counts().sort_index()/len(df)).plot(kind='barh')

plt.title('Tenure Distribution')

plt.xlabel('% Distribution');
sns.boxplot(x="TENURE", y="BALANCE", data=df)

plt.ylim(-10**3, 10**4)

plt.title('Balance distribution with Tenure');
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))

for i in range(4):

    for j in range(4):

        sns.distplot(df[df.columns[4 * i + j]], ax=axs[i,j])

plt.show()
df.shape
from sklearn.cluster import KMeans

k = 5

kmeans = KMeans(n_clusters=k, random_state=1)

df['k_5_label'] = kmeans.fit_predict(df)
kmeans.inertia_
profile = df.groupby('k_5_label').mean().T
round(profile)
# round(profile.apply(lambda x: (max(x) - min(x))/x.median(), axis=1))
round(pd.DataFrame(kmeans.cluster_centers_.T))
from sklearn.cluster import MiniBatchKMeans



minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=1)

df['k_5_batch'] = minibatch_kmeans.fit_predict(df)
pd.crosstab(df['k_5_label'], df['k_5_batch'])
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
def evaluate_metrics(df, min_clust=2, max_clust=10, rand_state=1):

    inertias = []

    silhouette = []

    ch_score = []

    db_score = []

    for n_clust in range(min_clust, max_clust):

        kmeans = KMeans(n_clusters=n_clust, random_state=rand_state)

        y_label = kmeans.fit_predict(df)

        inertias.append(kmeans.inertia_)

        silhouette.append(silhouette_score(df, y_label))

        ch_score.append(calinski_harabasz_score(df, y_label))

        db_score.append(davies_bouldin_score(df, y_label))        



    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0][0].plot(range(min_clust, max_clust), inertias, '-x', linewidth=2)

    ax[0][0].set_xlabel('No. of clusters')

    ax[0][0].set_ylabel('Inertia')

    

    ax[0][1].plot(range(min_clust, max_clust), silhouette, '-x', linewidth=2)

    ax[0][1].set_xlabel('No. of clusters')

    ax[0][1].set_ylabel('Silhouette Score')

    

    ax[1][0].plot(range(min_clust, max_clust), ch_score, '-x', linewidth=2)

    ax[1][0].set_xlabel('No. of clusters')

    ax[1][0].set_ylabel('Calinski Harabasz Score')

    

    ax[1][1].plot(range(min_clust, max_clust), db_score, '-x', linewidth=2)

    ax[1][1].set_xlabel('No. of clusters')

    ax[1][1].set_ylabel('Davies Bouldin Score')

    fig.suptitle('Metrics to evaluate the number of clusters')

    plt.show()
evaluate_metrics(df.iloc[:, :-2], min_clust=2, max_clust=15, rand_state=0)
df = df_original.copy()

df.fillna(df.median(), inplace=True)
from sklearn.preprocessing import StandardScaler

df_scaled = StandardScaler().fit_transform(df)
evaluate_metrics(df_scaled, min_clust=2, max_clust=15, rand_state=0)
from yellowbrick.cluster.silhouette import SilhouetteVisualizer
plt.style.use('seaborn-paper')

fig, axs = plt.subplots(2, 3, figsize=(20, 15))

axs = axs.reshape(6)

for i, k in enumerate(range(7, 13)):

    ax = axs[i]

    sil = SilhouetteVisualizer(KMeans(n_clusters=k, random_state=1), ax=ax)

    sil.fit(df_scaled)

    sil.finalize()
plt.style.use('fivethirtyeight')
df.T
kmeans = MiniBatchKMeans(n_clusters=8, random_state=1)

df['k_8_label'] = kmeans.fit_predict(df)
round(1e2 * df['k_8_label'].value_counts().sort_index()/len(df), 2)
round(df.groupby('k_8_label').mean().T, 2)
#fig, ax = plt.subplots(figsize=(6, 4))

df.mean()
round(1e2 * df['k_8_label'].value_counts().sort_index()/len(df))
(df[['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'k_8_label']]

 .groupby('k_8_label').mean().plot.bar(figsize=(15, 5)))

plt.title('Purchase Behavior of various segments')

plt.xlabel('SEGMENTS');
(df[['PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'k_8_label']]

 .groupby('k_8_label').mean().plot.bar(figsize=(15, 5)))

plt.title('Frequency behavior of various segments')

plt.xlabel('SEGMENTS');