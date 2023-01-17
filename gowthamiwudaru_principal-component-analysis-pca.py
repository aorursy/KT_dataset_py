## RUN THIS CELL TO PROPERLY HIGHLIGHT THE HEADER

import requests

from IPython.core.display import HTML

styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text

HTML(styles)
import os

import warnings

import numpy as np

import random

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

from scipy.sparse.linalg import eigs

import seaborn as sns

from IPython.display import display



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



%matplotlib inline



warnings.filterwarnings("ignore")
food_all = pd.read_csv('/kaggle/input/intro-to-ml/food-data.csv', index_col=0,sep=',').dropna(axis=0)



display(food_all)
### One-dimensional view (along the 'Real Coffee')

food_all['y'] =np.zeros(16) 



x = 'Real coffee'

y = 'y'



fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.regplot(x=x,

                 y=y,

                 data=food_all,

                 fit_reg=True,

                 marker="o",

                 color="skyblue")



# add annotations one by one with a loop

for i in range(food_all.shape[0]):

    label = food_all.index[i]

    x_pos = food_all[x][i]

    y_pos = food_all[y][i]



    ax.text(x_pos,

            y_pos,

            label,

            rotation=60,

            size='medium',

            color='black',

            weight='semibold')
### Two-dimensional view (along the 'Real Coffee' & 'Tea')

x = 'Real coffee'

y = 'Tea'



fig, ax = plt.subplots(figsize=(10, 5))

ax = sns.regplot(x=x,

                 y=y,

                 data=food_all,

                 fit_reg=False,

                 marker="o",

                 color="skyblue",

                 scatter_kws={'s': 350})



# add annotations one by one with a loop

for i in range(food_all.shape[0]):

    label = food_all.index[i]

    x_pos = food_all[x][i]

    y_pos = food_all[y][i]



    ax.text(x_pos,

            y_pos,

            label,

            horizontalalignment='left',

            size='medium',

            color='black',

            weight='semibold')
### Three-dimensional view (along the 'Real Coffee', 'Tea' & Potatoes)

m = food_all[['Real coffee', 'Tea', 'Potatoes']].values



fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')



for i in range(len(m)):

    x = m[i, 0]

    y = m[i, 1]

    z = m[i, 2]

    label = i

    ax.scatter(x, y, z, c='skyblue', s=60, alpha=1)

    ax.text(x,

            y,

            z,

            '%s' % (food_all.index[i]),

            horizontalalignment='left',

            size='medium',

            weight='semibold',

            zorder=1,

            color='black')



ax.set_xlabel('Real coffee')

ax.set_ylabel('Tea')

ax.set_zlabel('Potatoes')



plt.show()
### Four-dimensional view (along the 'Real Coffee', 'Tea' & Potatoes)

m = food_all[['Real coffee', 'Tea', 'Potatoes','Frozen fish']].values



fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')



img=ax.scatter(m[:,0],m[:,1], m[:,2],c=m[:,3], label=food_all.index,cmap=plt.hot(),s=60,alpha=1)

fig.colorbar(img)



ax.set_xlabel('Real coffee')

ax.set_ylabel('Tea')

ax.set_zlabel('Potatoes')

for i in range(len(m)):

    ax.text( m[i,0], m[i,1],m[i,2],food_all.index[i])

plt.show()
# Load dataset

nba_data = pd.read_csv('/kaggle/input/intro-to-ml/nba.csv', index_col=0, sep=',').dropna(axis=0)

# Reset index

nba_data.reset_index(drop=True, inplace=True)

nba_data.set_index('Player', inplace=True)



# Select numeric columns for PCA

nba_data = nba_data.select_dtypes(include=[np.number])



# dimensions

m, k = nba_data.shape



print("{} x {} table of data:".format(m, k))

display(nba_data.head())

print("...")
# First 250 players (alphabetically)

x = nba_data.head(250).values

x = x / x.std(axis=0)
x_centered = x - x.mean(axis=0)
C = x_centered.T.dot(x_centered) / x_centered.shape[0]

print(C[:4, :4])

print('...')

print(C.shape)
K = 2

l, W = eigs(C, k=K, v0=np.ones(25))



eigenvectors = pd.DataFrame(W.real,

                            columns=['Eigenvector 1', 'Eigenvector 2'],

                            index=nba_data.columns)



eigenvectors
eigenvalues = pd.DataFrame(l.real, index=['λ1', 'λ2'])

eigenvalues.plot(kind='bar',

                 title='First two Eigenvalues',

                 rot=0,

                 legend=False)
PCs = {

    'PC' + str(1 + i): (np.dot(x_centered, W[:, i]) / np.sqrt(l[i])).real

    for i in range(W.shape[1])

}



nba_pca = pd.DataFrame(PCs, index=nba_data.head(250).index)



x = 'PC1'

y = 'PC2'



fig, ax = plt.subplots(figsize=(25, 25))

ax = sns.regplot(data=nba_pca,

                 x=x,

                 y=y,

                 fit_reg=False,

                 marker="o",

                 color="skyblue",

                 scatter_kws={'s': 250})



# add annotations one by one with a loop

for i in range(nba_pca.shape[0]):

    label = nba_pca.index[i]

    x_pos = nba_pca[x][i] + .06

    y_pos = nba_pca[y][i] + .08



    ax.text(x_pos,

            y_pos,

            label,

            horizontalalignment='left',

            size='large',

            color='black',

            weight='semibold')
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

ax = sns.scatterplot(data=nba_pca,

                     x=x,

                     y=y,

                     marker="o",

                     s=250,

                     hue=(nba_data['MP']),

                     ax=ax)



for i in range(nba_pca.shape[0]):

    label = nba_pca.index[i]

    x_pos = nba_pca[x][i] + .06

    y_pos = nba_pca[y][i] + .08



    ax.text(x_pos,

            y_pos,

            label,

            horizontalalignment='left',

            size='medium',

            color='black',

            weight='semibold')

plt.show()
from sklearn.cluster import KMeans

X=nba_pca

colors = ['black', 'blue', 'purple', 'yellow', 'red']

kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

fig, ax = plt.subplots(1, 1, figsize=(15, 15))

x = 'PC1'

y = 'PC2'

ax = sns.scatterplot(data=nba_pca,

                     x=x,

                     y=y,

                     marker="o",

                     s=250,

                     hue=(kmeans.predict(X)),

                     palette=sns.color_palette("hls", 5),

                     legend="full",

                     ax=ax)



for i in range(nba_pca.shape[0]):

    label = nba_pca.index[i]

    x_pos = nba_pca[x][i] + .06

    y_pos = nba_pca[y][i] + .08



    ax.text(x_pos,

            y_pos,

            label,

            horizontalalignment='left',

            size='medium',

            color='black',

            weight='semibold')

plt.show()
from sklearn.datasets import load_digits

from sklearn.decomposition import PCA



# loading dataset

digits = load_digits()

X = digits.data / 255.0

y = digits.target



# transforming into a Pandas dataframe

feat_cols = ['pixel_' + str(i) for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=feat_cols)

df['label'] = y

df['label'] = df['label'].apply(lambda i: str(i))

X, y = None, None



print("{} x {} table of data:".format(df.shape[0], df.shape[1]))

display(df.head())

print("...")
np.random.seed(42)

rndperm = np.random.permutation(df.shape[0])



plt.gray()

fig = plt.figure(figsize=(8, 6))

for i in range(0, 15):

    ax = fig.add_subplot(3,

                         5,

                         i + 1,

                         title="\nDigit: {}\n".format(

                             str(df.loc[rndperm[i], 'label'])))

    ax.grid(False)

    ax.set_xticks([])

    ax.set_yticks([])

    ax.matshow(df.loc[rndperm[i], feat_cols].values.reshape(

        (8, 8)).astype(float))

plt.show()
# Count per label

df.label.value_counts()
pca = PCA(n_components=25)

pca_result = pca.fit_transform(df[feat_cols].values)



for i in range(25):

    df['PC' + str(1 + i)]= pca_result[:,i]



plt.figure(figsize=(16, 10))

sns.scatterplot(x="PC1",

                y="PC2",

                hue="label",

                palette=sns.color_palette("hls", 10),

                data=df.loc[rndperm, :],

                legend="full",

                alpha=1)
plt.figure(figsize=(16, 10))

g = sns.relplot(x="PC1", y="PC2",

                 col="label", hue="PC3",

                  col_wrap=4,data=df)
np.random.seed(42)

rndperm = np.random.permutation(df.shape[0])

columns=[]

for i in range(25):

    columns.append('PC' + str(1 + i))

plt.gray()

fig = plt.figure(figsize=(8, 6))

for i in range(0, 15):

    ax = fig.add_subplot(3,

                         5,

                         i + 1,

                         title="\nDigit: {}\n".format(

                             str(df.loc[rndperm[i], 'label'])))

    ax.grid(False)

    ax.set_xticks([])

    ax.set_yticks([])

    ax.matshow(df.loc[rndperm[i], columns[:4]].values.reshape(

        (2, 2)).astype(float))

plt.show()
#scree plot

pca = PCA(n_components=64)

pca_result = pca.fit_transform(df[feat_cols].values)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')
print(np.cumsum(pca.explained_variance_ratio_))
#reconstruct from 10 components

pca = PCA(n_components=10)

pca_result = pca.fit_transform(df[feat_cols].values)

reconst=pca.inverse_transform(pca_result)

columns=[]

for i in range(64):

    df['RC' + str(1 + i)]= reconst[:,i]

    columns.append('RC' + str(1 + i))

fig = plt.figure(figsize=(8, 6))

for i in range(0, 15):

    ax = fig.add_subplot(3,

                         5,

                         i + 1,

                         title="\nDigit: {}\n".format(

                             str(df.loc[rndperm[i], 'label'])))

    ax.grid(False)

    ax.set_xticks([])

    ax.set_yticks([])

    ax.matshow(df.loc[rndperm[i], columns].values.reshape(

        (8, 8)).astype(float))

plt.show()
#reconstruct from 40 components

pca = PCA(n_components=40)

pca_result = pca.fit_transform(df[feat_cols].values)

reconst=pca.inverse_transform(pca_result)

columns=[]

for i in range(64):

    df['RC' + str(1 + i)]= reconst[:,i]

    columns.append('RC' + str(1 + i))

fig = plt.figure(figsize=(8, 6))

for i in range(0, 15):

    ax = fig.add_subplot(3,

                         5,

                         i + 1,

                         title="\nDigit: {}\n".format(

                             str(df.loc[rndperm[i], 'label'])))

    ax.grid(False)

    ax.set_xticks([])

    ax.set_yticks([])

    ax.matshow(df.loc[rndperm[i], columns].values.reshape(

        (8, 8)).astype(float))

plt.show()