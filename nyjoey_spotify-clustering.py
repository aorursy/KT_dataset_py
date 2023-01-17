# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.cluster import DBSCAN

from sklearn.cluster import AgglomerativeClustering

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import silhouette_score

InteractiveShell.ast_node_interactivity = "all"

%matplotlib inline
df = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='latin1')

df.head()

df.shape
df = df.rename(columns = {'Unnamed: 0': 'id'})

df.head()
df.isnull().any()
df = df.drop_duplicates()

df.shape
df = df.drop(['id'], axis=1)

df.head()
df['top genre'].value_counts().head()

df['artist'].value_counts().head()

df['title'].value_counts().head()

df['year'].value_counts().head()
df[df.title == 'Company']
yearless_df = df.drop(['year', 'title', 'pop'], axis=1)

yearless_df.drop_duplicates()
df['top genre'].value_counts()
for i in yearless_df['top genre']:

    if 'pop' in i:

        yearless_df['top genre'] = yearless_df['top genre'].replace(i, 'pop')

        

    elif 'hip hop' in i:

        yearless_df['top genre'] = yearless_df['top genre'].replace(i, 'hip hop')



    elif 'edm' in i:

        yearless_df['top genre'] = yearless_df['top genre'].replace(i, 'edm')



    elif 'r&b' in i:

        yearless_df['top genre'] = yearless_df['top genre'].replace(i, 'pop')



    elif 'latin' in i:

        yearless_df['top genre'] = yearless_df['top genre'].replace(i, 'latin')



    elif 'room' in i:

        yearless_df['top genre'] = yearless_df['top genre'].replace(i, 'room')



    elif 'electro' in i:

        yearless_df['top genre'] = yearless_df['top genre'].replace(i, 'edm')

        

yearless_df['top genre'] = yearless_df['top genre'].replace('chicago rap', 'hip hop')

        

yearless_df["top genre"]
yearless_df['top genre'].value_counts()
yearless_df
# genre_df = pd.DataFrame(yearless_df['top genre'].value_counts()).reset_index()

# genre_df.columns = ['top genre','count']

# genre_df['top_genre_modeling'] = genre_df['top genre'] 

# genre_df.loc[genre_df['count']< 4,'top_genre_modeling'] = 'other'

# genre_df = genre_df.drop(['top genre'], axis=1)

# genre_df
temp_df = yearless_df

value_counts = temp_df.stack().value_counts() # Entire DataFrame 

to_remove = value_counts[value_counts <= 3].index

temp_df.replace(to_remove, 'other', inplace=True)

temp_df['top genre'].value_counts()

temp_df.head()

temp_df.shape
yearless_df['top genre'] = temp_df['top genre']

yearless_df[['bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch', 'artist']] = df[['bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch', 'artist']]

yearless_df['top genre'].value_counts()

yearless_df.head()
new_df = yearless_df

new_df.artist.unique()
new_df.isnull().any()

new_df = new_df.drop_duplicates()

new_df = new_df.reset_index(drop=True)

new_df.head()
plt.hist(new_df.dB)

plt.show()

plt.hist(new_df.bpm)

plt.show()

plt.hist(new_df.nrgy)

plt.show()

plt.hist(new_df.live)

plt.show()

plt.hist(new_df.val)

plt.show()

plt.hist(new_df.dur)

plt.show()

plt.hist(new_df.acous)

plt.show()

plt.hist(new_df.spch)

plt.show()
new_df
new_df.bpm.unique()

new_df.dB.unique()
new_df.bpm = new_df.bpm.replace(0, new_df.bpm.mean())

new_df.bpm.unique()

new_df.dB = new_df.dB.replace(-60, new_df.dB.mean())

new_df.dB.unique()
temp_df = pd.get_dummies(new_df[['artist', 'top genre']])

new_df = new_df.join(temp_df, how='left')

new_df = new_df.drop(columns = ['artist', 'top genre'], axis=1)

new_df.shape
X_std = StandardScaler().fit_transform(new_df)

pca = PCA(n_components=.95)

principalComponents = pca.fit_transform(X_std) # Plot the explained variances



#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Pulsar Dataset Explained Variance')

plt.show()



# plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')

# plt.xlabel('PCA 1')

# plt.ylabel('PCA 2')

# PCA_components = pd.DataFrame(principalComponents)
pca = PCA(n_components=20)

principalComponents = pca.fit_transform(X_std)

pca_df = pd.DataFrame(principalComponents)
sum_of_squared_distances = []

K = range(1,20)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(pca_df)

    sum_of_squared_distances.append(km.inertia_)

    

ax = sns.lineplot(x=K, y = sum_of_squared_distances)

ax.set(xlabel='K', ylabel='sum of squared distances', title='Elbow graph')
kmeans = KMeans(n_clusters=13)    

kmeans.fit(pca_df)

y_kmeans = kmeans.predict(pca_df)

y_kmeans
plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.figure(figsize=(20, 10))

plt.title("Spotify Dendograms")

dendogram = dendrogram(linkage(pca_df, method='ward'))
ac = AgglomerativeClustering(n_clusters=13, affinity='euclidean', linkage='ward')

ac.fit_predict(pca_df)
dbscan = DBSCAN(eps = 9, min_samples = 3)

dbscan.fit(pca_df)

labels = dbscan.labels_



# Number of clusters in labels, ignoring noise if present.

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
kmeans_labels = pd.DataFrame(kmeans.labels_)

ac_labels = pd.DataFrame(ac.labels_)

dbscan_labels = pd.DataFrame(dbscan.labels_)



silhouette_score(pca_df, kmeans_labels, metric='euclidean')

silhouette_score(pca_df, ac_labels, metric='euclidean')

silhouette_score(pca_df, dbscan_labels, metric='euclidean')
dbscan_df = new_df.join(dbscan_labels, how='left')

dbscan_df = dbscan_df.rename(columns = {0: 'labels'})

dbscan_df.head()
df_scaled = pd.DataFrame(new_df)

df_scaled['dbscan'] = dbscan.labels_

df_mean = (df_scaled.loc[df_scaled.dbscan!=-1, :].groupby('dbscan').mean())

results = pd.DataFrame(columns=['Variable', 'Var'])

for column in df_mean.columns[1:]:

    results.loc[len(results), :] = [column, np.var(df_mean[column])]

    selected_columns = list(results.sort_values('Var', ascending=False,).head(7).Variable.values) + ['dbscan']

    tidy = df_scaled[selected_columns].melt(id_vars='dbscan')
# 12 is the number of clusters in DBScan

for i in range(12):

    sns.catplot(x='dbscan', y='value', hue='variable', data=tidy[tidy['dbscan']==i], height=5, aspect=.7, kind='bar')