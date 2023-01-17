import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/german-credit/german_credit_data.csv', index_col=0)
df.head()
df.shape
df.isnull().sum()
df['Saving accounts'] = df['Saving accounts'].fillna('None')

df['Saving accounts'].value_counts(normalize=True)
df['Checking account'] = df['Checking account'].fillna('None')

df['Checking account'].value_counts(normalize=True)
df.head()
df.shape
df.Sex.hist();
df['Sex'] = df['Sex'].apply(lambda x: 1 if x=='male' else 0)
df.Housing.value_counts(normalize=True)
df.Purpose.value_counts(normalize=True)
df['Purpose'].replace(['repairs', 'domestic appliances', 'vacation/others'], 'others', inplace=True)
df.Purpose.value_counts(normalize=True)
df['Saving accounts'].value_counts(normalize=True)
df['Saving accounts'].replace(['None', 'little', 'moderate', 'rich', 'quite rich'], [0,1,2,3,4], inplace=True)
df['Checking account'].value_counts(normalize=True)
df['Checking account'].replace(['None', 'little', 'moderate', 'rich'], [0,1,2,3], inplace=True)
df.head()
df.hist(figsize=(12,12));
df['Duration'] = np.log(df['Duration'])

df['Age'] = np.log(df['Age'])

df['Credit amount'] = np.log(df['Credit amount'])
sns.pairplot(df);
plt.figure(figsize=(20,20))

sns.heatmap(df.corr(), annot=True);
df = pd.get_dummies(df, drop_first=True)
df.head()
df.shape
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN



from scipy.cluster import hierarchy

from scipy.spatial.distance import pdist



from sklearn.metrics import silhouette_score

from sklearn.manifold import TSNE
inertia = []

k = range(1, 11)

for k_i in k:

    km = KMeans(n_clusters=k_i).fit(df)

    inertia.append(km.inertia_)

    

plt.plot(k, inertia)

plt.xlabel('k')

plt.ylabel('inertia')

plt.title('The Elbow Method showing the optimal k');
distance_mat = pdist(df)



Z = hierarchy.linkage(distance_mat, 'ward')
plt.figure(figsize=(20, 10))



plt.title('Hierarchical Clustering Dendrogram (truncated)')

plt.xlabel('cluster size')

plt.ylabel('distance')

hierarchy.dendrogram(

    Z,

    truncate_mode='lastp',

    p=12,  

    leaf_font_size=12.,

    show_contracted=True, 

)

plt.show()
silhouette_scores = [] 

k = range(2,8)



for n_cluster in k:

    silhouette_scores.append( 

        silhouette_score(df, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(df))) 

    

    

# Plotting a bar graph to compare the results 



plt.bar(k, silhouette_scores) 

plt.xlabel('Number of clusters', fontsize = 10) 

plt.ylabel('Silhouette Score', fontsize = 10) 

plt.show() 
db = DBSCAN(eps=1.61, min_samples=4).fit(df)
# Number of clusters in labels, ignoring noise if present.

n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

n_noise_ = list(db.labels_).count(-1)



print('Estimated number of clusters: {}'.format(n_clusters_))

print('Estimated percentage of noise points: {:.2f}%'.format(100*n_noise_/df.shape[0]))
from sklearn.manifold import TSNE
def draw_tsne(df):

    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharey=True)



    tsne=TSNE(perplexity=5).fit_transform(df)

    axes[0, 0].title.set_text('Perplexity 5')

    sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], ax=axes[0, 0]);



    tsne=TSNE(perplexity=10).fit_transform(df)

    axes[0, 1].title.set_text('Perplexity 10')

    sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], ax=axes[0, 1]);



    tsne=TSNE(perplexity=20).fit_transform(df)

    axes[0, 2].title.set_text('Perplexity 20')

    sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], ax=axes[0, 2]);



    tsne=TSNE(perplexity=30).fit_transform(df)

    axes[1, 0].title.set_text('Perplexity 30')

    sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], ax=axes[1, 0]);



    tsne=TSNE(perplexity=40).fit_transform(df)

    axes[1, 1].title.set_text('Perplexity 40')

    sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], ax=axes[1, 1]);



    tsne=TSNE(perplexity=50).fit_transform(df)

    axes[1, 2].title.set_text('Perplexity 50')

    sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], ax=axes[1, 2]);
draw_tsne(df)
tsne=TSNE(perplexity=30).fit_transform(df)
plt.figure(figsize=(12,12))

plt.title('Perplexity 30')

sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1]);
plt.figure(figsize=(10, 10))

plt.title('DBSCAN, 2 clusters')

plt.scatter(tsne[:, 0], tsne[:, 1], c=db.labels_);
km = KMeans(n_clusters=2).fit(df)

agg_cluster = AgglomerativeClustering(n_clusters = 2).fit(df)



_, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharey=True)



axes[0].title.set_text('K-MEANS, 2 clusters')

sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], hue=km.labels_, ax=axes[0]);





plt.title('Hierarchical clustering, 2 clusters')

sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], hue=agg_cluster.labels_, ax=axes[1]);
km = KMeans(n_clusters=3).fit(df)

agg_cluster = AgglomerativeClustering(n_clusters = 3).fit(df)



_, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharey=True)



axes[0].title.set_text('K-MEANS, 3 clusters')

sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], hue=km.labels_, ax=axes[0], palette=['green','orange','brown']);





plt.title('Hierarchical clustering, 3 clusters')

sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], hue=agg_cluster.labels_, ax=axes[1], palette=['green','orange','brown']);
km = KMeans(n_clusters=4).fit(df)

agg_cluster = AgglomerativeClustering(n_clusters = 4).fit(df)



_, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharey=True)



axes[0].title.set_text('K-MEANS, 4 clusters')

sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], hue=km.labels_, ax=axes[0], palette=['green','orange','brown', 'yellow']);





plt.title('Hierarchical clustering, 4 clusters')

sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], hue=agg_cluster.labels_, ax=axes[1], palette=['green','orange','brown', 'yellow']);
agg_cluster = AgglomerativeClustering(n_clusters = 3).fit(df)
fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(40, 20))



i_col = 0

i_row = 0



for column in df.columns:

    sns.boxplot(y=column, x=agg_cluster.labels_, 

                     data=df, 

                     palette="colorblind", ax=ax[i_row, i_col])

    if i_row < 4:

        i_row += 1

    else:

        i_col += 1

        i_row = 0
