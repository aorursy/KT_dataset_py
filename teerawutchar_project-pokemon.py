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
# data analysis and wrangling

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn import preprocessing

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt



#import ข้อมูลเข้ามาใช้งาน

pokemondata = pd.read_csv('../input/pokemon/pokemon_alopez247.csv')

pokemondata1 = pd.read_csv('../input/pokemon/pokemon_alopez247.csv')

pokemon = pokemondata.copy()

pokemon1 = pokemondata1.copy()

pokemon.head()
pokemon.info()
pokemon.isnull().any()
Select_f = ['Type_1', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Height_m']

pokemon = pokemon[Select_f]

Select_f = ['Type_1', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Height_m']

pokemon1 = pokemon1[Select_f]

pokemon.head()
pokemon.describe()
sns.heatmap(pokemon.corr(), annot=True)
cols =['HP','Attack','Defense','Sp_Def','Speed','Height_m']
pt = preprocessing.PowerTransformer(method='yeo-johnson',standardize=True)

mat = pt.fit_transform(pokemon1[cols])

mat[:5].round(4)
X=pd.DataFrame(mat, columns=cols)

X.head()
fig, ax=plt.subplots(figsize=(20,8))

dg=sch.dendrogram(sch.linkage(X,method='ward'),ax=ax,labels=pokemon1['Type_1'].values)



sns.clustermap(X,col_cluster=False,cmap="Blues")
hc=AgglomerativeClustering(n_clusters=10,linkage='ward')

hc
hc.fit(X)
hc.labels_
pokemon1['cluster']=hc.labels_

pokemon1.head()
pokemon1.head(10)
##แสดงค่าMean

pokemon1.groupby('cluster').agg(['mean']).T
cols =['HP','Attack','Defense','Sp_Atk','Sp_Def','Speed','Height_m']



fig,ax = plt.subplots(nrows=4,ncols=2,figsize=(20,9))

ax=ax.ravel()

for i, col in enumerate(cols):

    sns.violinplot(x='cluster',y=col,data=pokemon1,ax=ax[i])
from sklearn.cluster import KMeans



sse = {}

for k in range(1, 21):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(pokemon.drop(columns=['Type_1']))

    pokemon["clusters"] = kmeans.labels_

    #print(data["clusters"])

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()
print(sse)

# for k in range(1,11):

#     print(str(k) + ': ' + str(sse[k] - sse[k+1]))
kmeans = KMeans(n_clusters=10, max_iter=2000).fit(pokemon.drop(columns=['Type_1']))

pokemon["clusters"] = kmeans.labels_

pokemon.groupby(by=['clusters']).mean()
sns.pairplot(pokemon, vars=['HP','Attack','Defense','Sp_Def','Speed','Height_m'],hue='clusters',plot_kws={'alpha': .4});
sns.catplot(x="clusters", y="HP", data=pokemon)
sns.catplot(x="clusters", y="Attack", data=pokemon)
sns.catplot(x="clusters", y="Defense", data=pokemon)
sns.catplot(x="clusters", y="Sp_Atk", data=pokemon)
sns.catplot(x="clusters", y="Sp_Def", data=pokemon)
sns.catplot(x="clusters", y="Speed", data=pokemon)
sns.catplot(x="clusters", y="Height_m", data=pokemon)