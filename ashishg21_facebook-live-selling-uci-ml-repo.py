import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from IPython.display import display

import os
warnings.filterwarnings('ignore') # ignore warnings.

%config IPCompleter.greedy = True # autocomplete feature.

pd.options.display.max_rows = None # set maximum rows that can be displayed in notebook.

pd.options.display.max_columns = None # set maximum columns that can be displayed in notebook.

pd.options.display.precision = 2 # set the precision of floating point numbers.
df = pd.read_csv('../input/Live.csv', encoding='utf-8')

df.drop_duplicates(inplace=True) # drop duplicates if any.

df.shape # num rows x num columns.
miss_val = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)

miss_val[miss_val>0]
df.drop(labels=['Column1', 'Column2', 'Column3','Column4'], axis=1, inplace=True)
df.head()
df.drop('status_id', axis=1, inplace=True)
df['status_type_isvideo'] = df['status_type'].map(lambda x:1 if(x=='video') else 0)

df.drop('status_type', axis=1, inplace=True)
df['status_published'] = pd.to_datetime(df['status_published'])
df['year'] = df['status_published'].dt.year

df['month'] = df['status_published'].dt.month

df['dayofweek'] = df['status_published'].dt.dayofweek # 0 is Monday, 7 is Sunday.

df['hour'] = df['status_published'].dt.hour
reaction = ['num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',

            'num_sads', 'num_angrys'] # reaction of users.
before2016 = df[df['year']<=2015]

after2016 = df[df['year']>2015]
before2016[reaction].describe()
after2016[reaction].describe()
before2016.groupby('status_type_isvideo')[reaction].mean()
sns.heatmap(before2016[reaction].corr(), cmap='coolwarm', annot=True)
from sklearn.preprocessing import StandardScaler



standard_scaler = StandardScaler()

before2016_s = before2016[reaction]

before2016_s = standard_scaler.fit_transform(before2016_s) # s in before2016_s stands for scaled.
# Improting the PCA module.



from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized', random_state=123)
# Doing the PCA on the data.

pca.fit(before2016_s)
# Making the screeplot - plotting the cumulative variance against the number of components



fig = plt.figure(figsize = (10,5))



plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')



plt.show()
# what percentage of variance in data can be explained by first 2,3 and 4 principal components respectively?

(pca.explained_variance_ratio_[0:2].sum().round(3),

pca.explained_variance_ratio_[0:3].sum().round(3),

pca.explained_variance_ratio_[0:4].sum().round(3))
# we'll use first 2 principal components to visualise feature importance.



loadings = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':reaction})

loadings
# we can visualize what the principal components seem to capture.



fig = plt.figure(figsize = (6,6))

plt.scatter(loadings.PC1, loadings.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

for i, txt in enumerate(loadings.Feature):

    plt.annotate(txt, (loadings.PC1[i],loadings.PC2[i]))

plt.tight_layout()

plt.show()
before2016.groupby('year').sum()[reaction]
before2016.groupby('year').sum()[reaction].plot(figsize=(12,5))
before2016.groupby(['year', 'status_type_isvideo']).sum()[reaction]
plt.figure(1)

before2016[before2016['status_type_isvideo']==0].groupby('year').sum()[reaction].plot(

    figsize=(10,5), title='photo content')



plt.figure(2)

before2016[before2016['status_type_isvideo']==1].groupby('year').sum()[reaction].plot(

    figsize=(10,5), title='video content')
plt.figure(1)

before2016[before2016['status_type_isvideo']==0].groupby('month').sum()[reaction].plot(

    figsize=(10,5), title='photo content')



plt.figure(2)

before2016[before2016['status_type_isvideo']==1].groupby('month').sum()[reaction].plot(

    figsize=(10,5), title='video content')
plt.figure(1)

before2016[before2016['status_type_isvideo']==0].groupby('dayofweek').sum()[reaction].plot(

    figsize=(10,5), title='photo content')



plt.figure(2)

before2016[before2016['status_type_isvideo']==1].groupby('dayofweek').sum()[reaction].plot(

    figsize=(10,5), title='video content')
plt.figure(1)

before2016[before2016['status_type_isvideo']==0].groupby('hour').sum()[reaction].plot(

    figsize=(10,5), title='photo content')



plt.figure(2)

before2016[before2016['status_type_isvideo']==1].groupby('hour').sum()[reaction].plot(

    figsize=(10,5), title='video content')
after2016.groupby('status_type_isvideo')[reaction].mean()
sns.heatmap(after2016[reaction].corr(), cmap='coolwarm', annot=True)
from sklearn.preprocessing import StandardScaler



standard_scaler = StandardScaler()

after2016_s = after2016[reaction]

after2016_s = standard_scaler.fit_transform(after2016_s) # s in before2016_s stands for scaled.
# Improting the PCA module.



from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized', random_state=123)
# Doing the PCA on the data.

pca.fit(after2016_s)
# Making the screeplot - plotting the cumulative variance against the number of components



fig = plt.figure(figsize = (10,5))



plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')



plt.show()
# what percentage of variance in data can be explained by first 2,3 and 4 principal components respectively?

(pca.explained_variance_ratio_[0:2].sum().round(3),

pca.explained_variance_ratio_[0:3].sum().round(3),

pca.explained_variance_ratio_[0:4].sum().round(3))
# we'll use first 2 principal components to visualise feature importance.



loadings = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':reaction})

loadings
# we can visualize what the principal components seem to capture.



fig = plt.figure(figsize = (6,6))

plt.scatter(loadings.PC1, loadings.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

for i, txt in enumerate(loadings.Feature):

    plt.annotate(txt, (loadings.PC1[i],loadings.PC2[i]))

plt.tight_layout()

plt.show()
after2016.groupby('year').sum()[reaction]
after2016.groupby('year').sum()[reaction].plot(figsize=(12,5))
after2016.groupby(['year', 'status_type_isvideo']).sum()[reaction]
plt.figure(1)

after2016[after2016['status_type_isvideo']==0].groupby('year').sum()[reaction].plot(

    figsize=(10,5), title='photo content')



plt.figure(2)

after2016[after2016['status_type_isvideo']==1].groupby('year').sum()[reaction].plot(

    figsize=(10,5), title='video content')
plt.figure(1)

after2016[after2016['status_type_isvideo']==0].groupby('month').sum()[reaction].plot(

    figsize=(10,5), title='photo content')



plt.figure(2)

after2016[after2016['status_type_isvideo']==1].groupby('month').sum()[reaction].plot(

    figsize=(10,5), title='video content')
plt.figure(1)

after2016[after2016['status_type_isvideo']==0].groupby('dayofweek').sum()[reaction].plot(

    figsize=(10,5), title='photo content')



plt.figure(2)

after2016[after2016['status_type_isvideo']==1].groupby('dayofweek').sum()[reaction].plot(

    figsize=(10,5), title='video content')
plt.figure(1)

after2016[after2016['status_type_isvideo']==0].groupby('hour').sum()[reaction].plot(

    figsize=(10,5), title='photo content')



plt.figure(2)

after2016[after2016['status_type_isvideo']==1].groupby('hour').sum()[reaction].plot(

    figsize=(10,5), title='video content')