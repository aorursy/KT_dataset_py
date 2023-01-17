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

color = sns.color_palette()

sns.set(rc={'figure.figsize':(25,15)})

from IPython.core.interactiveshell import InteractiveShell

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from scipy.cluster.hierarchy import linkage, dendrogram

import scipy.cluster.hierarchy as sch

from sklearn.cluster import DBSCAN

from sklearn.cluster import AgglomerativeClustering

import statsmodels.api as sm

from sklearn import preprocessing

from statsmodels.stats.outliers_influence import variance_inflation_factor



from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import silhouette_score

InteractiveShell.ast_node_interactivity = "all"

%matplotlib inline
df = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='latin1')

df1 = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='latin1')

df.head()

df.shape
df = df.rename(columns = {'Unnamed: 0': 'id'})

df.head()
df.isnull().any()
df.isnull().sum()
df = df.drop_duplicates()

df.shape
df = df.drop(['id'], axis=1)

df.head()
df.describe().T
df['top genre'].value_counts()

df['artist'].value_counts()

df['title'].value_counts()

df['year'].value_counts()
df[df.title == 'Company']
song=df[df["pop"]==df["pop"]].groupby(["artist"]).agg({"pop":'max',"dnce":'max',"spch":'max'}).sort_values(["pop"],ascending=False)

print(song)
raphit=song[(song["pop"]>=90)&(song["spch"]>=30)]

print (raphit)
yearless_df = df.drop(['year', 'title','dB','live','dur'], axis=1)

yearless_df.drop_duplicates()
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
sns.pairplot(yearless_df, vars=['bpm','nrgy','dnce','val','acous','spch','pop'],

            hue='top genre'

            

           );
hit=df[df["pop"]==df["pop"]].groupby(["title"]).agg({"pop":'max',"dnce":'max',"spch":'max',"bpm":'max'}).sort_values(["pop"],ascending=False)

print(hit)
pop=df[df["pop"]==df["pop"]].groupby(["year"]).agg({"pop":'max',"dnce":'max',"spch":'max',"bpm":'max'}).sort_values(["pop"],ascending=False)

print(pop)
plot_data =yearless_df.groupby('top genre')['pop'].max()



plot_data.sort_values()[-10:].plot(kind='bar')

plt.title("Popurality Genre")

plt.ylabel("Popurality")

plt.show()
plt.hist(yearless_df.dB)

plt.xlabel('Loudness dB')

plt.show()

plt.hist(yearless_df.bpm)

plt.xlabel('Beat per minute')

plt.show()

plt.hist(yearless_df.nrgy)

plt.xlabel('Energy')

plt.show()

plt.hist(yearless_df.live)

plt.xlabel('Liveness')

plt.show()

plt.hist(yearless_df.val)

plt.xlabel('Valence')

plt.show()

plt.hist(yearless_df.dur)

plt.xlabel('Durable')

plt.show()

plt.hist(yearless_df.acous)

plt.xlabel('Acousticness')

plt.show()

plt.hist(yearless_df.spch)

plt.xlabel('Speechiness')

plt.show()
dfh = yearless_df.drop(['live','dB','dur'], axis=1)

dfk = yearless_df.drop(['live','dB','dur'], axis=1)
cols =['bpm','nrgy','dnce','val','acous','spch','pop']
pt = preprocessing.PowerTransformer(method='yeo-johnson',standardize=True)

mat = pt.fit_transform(dfh[cols])

mat[:5].round(4)
X=pd.DataFrame(mat, columns=cols)

X.head()
fig, ax=plt.subplots(figsize=(100,20))

dg=sch.dendrogram(sch.linkage(X,method='ward'),ax=ax,labels=dfh['top genre'].values)
hc=AgglomerativeClustering(n_clusters=8,linkage='ward')

hc
hc.fit(X)
hc.labels_
dfh['cluster']=hc.labels_

dfh.head()
dfh.head(10)
dfh.groupby('cluster').agg(['count','mean','median']).T
dfh.groupby('cluster').head(3).sort_values('cluster')
cols =['bpm','nrgy','dnce','val','acous','spch','pop']

fig,ax = plt.subplots(nrows=4,ncols=2,figsize=(20,9))

ax=ax.ravel()

for i, col in enumerate(cols):

    sns.violinplot(x='cluster',y=col,data=dfh,ax=ax[i])
data = dfh.iloc[:, 3:5].values
plt.figure(figsize=(10, 7))

plt.scatter(data[:,0], data[:,1], c=dfh.cluster, cmap='rainbow')
pd.crosstab(dfh['top genre'],hc.labels_)
X = dfk[['bpm','nrgy','dnce','val','acous','spch','pop']]
sum_of_squared_distances = []

K = range(1,20)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(X)

    sum_of_squared_distances.append(km.inertia_)

    

ax = sns.lineplot(x=K, y = sum_of_squared_distances)

ax.set(xlabel='K', ylabel='sum of squared distances', title='Elbow graph')
model = KMeans(n_clusters=8)

model
model.fit(X)
model.cluster_centers_
model.labels_
dfk['top genre'].values
dfk['cluster']=model.labels_
pd.crosstab(dfk['top genre'],model.labels_)
dfk.head(10)
dfk.groupby('cluster').head(3).sort_values('cluster')
model.predict([

    [65,88,70,50,3,13,75],

    [120,50,70,70,2,15,75]

])
sns.catplot(x="cluster",y="bpm",data=dfh)

plt.xlabel('cluster H')

sns.catplot(x="cluster",y="bpm",data=dfk)

plt.xlabel('cluster K')
sns.catplot(x="cluster",y="nrgy",data=dfh)

plt.xlabel('cluster H')

sns.catplot(x="cluster",y="nrgy",data=dfk)

plt.xlabel('cluster K')
sns.catplot(x="cluster",y="dnce",data=dfh)

plt.xlabel('cluster H')

sns.catplot(x="cluster",y="dnce",data=dfk)

plt.xlabel('cluster K')
sns.catplot(x="cluster",y="val",data=dfh)

plt.xlabel('cluster H')

sns.catplot(x="cluster",y="val",data=dfk)

plt.xlabel('cluster K')
sns.catplot(x="cluster",y="acous",data=dfh)

plt.xlabel('cluster H')

sns.catplot(x="cluster",y="acous",data=dfk)

plt.xlabel('cluster K')
sns.catplot(x="cluster",y="spch",data=dfh)

plt.xlabel('cluster H')

sns.catplot(x="cluster",y="spch",data=dfk)

plt.xlabel('cluster K')
sns.catplot(x="cluster",y="pop",data=dfh)

plt.xlabel('cluster H')

sns.catplot(x="cluster",y="pop",data=dfk)

plt.xlabel('cluster K')