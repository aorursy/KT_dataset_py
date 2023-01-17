# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np
responses = pd.read_csv( "../input/responses.csv")

responses.head()
df = pd.DataFrame(responses)
df['House - block of flats'] = np.where(df['House - block of flats']=="block of flats",1,2)



df['Village - town'] = np.where(df['Village - town']=="village",1,2)



df['Left - right handed'] = np.where(df['Left - right handed']=="right handed",1,2)



df['Only child'] = np.where(df['Only child']=="yes",1,2)



df['Gender'] = np.where(df['Gender']=="male",1,2) #male=1, female=2



df['Internet usage'] = df['Internet usage'].astype('category')

df['Internet usage'] = df['Internet usage'].cat.codes + 1



df['Punctuality'] = df['Punctuality'].astype('category')

df['Punctuality'] = df['Punctuality'].cat.codes + 1



df['Lying'] = df['Lying'].astype('category')

df['Lying'] = df['Lying'].cat.codes + 1



df['Smoking'] = df['Smoking'].astype('category')

df['Smoking'] = df['Smoking'].cat.codes + 1



df['Alcohol'] = df['Alcohol'].astype('category')

df['Alcohol'] = df['Alcohol'].cat.codes + 1
df.head()
def cat_to_num(x):

    if x=="currently a primary school pupil":

        return 1

    if x=="primary schoool":

        return 2

    if x=="secondary school":

        return 3

    if x=="college/bachelor degree":

        return 4

    if x=="masters degree ":

        return 5

    if x=="doctorate degree ":

        return 6

    

    

df['Education'] = df['Education'].apply(cat_to_num)
df.head()
df1 = pd.DataFrame(df)

gen_df = df1.iloc[:,76:133]
import missingno as msno



msno.matrix(gen_df)

msno.heatmap(gen_df)
def myMedianimpute(data):

    for i in data.columns:

        data[i] = data[i].replace('?',np.nan).astype(float)

        data[i] = data[i].fillna((data[i].median()))

    return data



myMedianimpute(gen_df)
np.sum(gen_df.isna())
df2 = pd.concat([gen_df,df1.Gender],axis=1)

df2.columns
len(gen_df.columns)
import seaborn as sns

import matplotlib.pyplot as plt



for i in  df2.columns:

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    sns.countplot(y=i, data=df2, ax=ax[0])

    sns.countplot(y=i, hue='Gender', data=df2, ax=ax[1])

    plt.xticks(fontsize=14)

    plt.yticks(fontsize=14)
import matplotlib.pyplot as plt

from scipy.stats import spearmanr



spr_corr = gen_df.corr(method='spearman')



fig = plt.figure(figsize=(25,10))

ax = fig.add_subplot(111)

cax = ax.matshow(spr_corr,cmap='coolwarm',vmin=-1,vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(gen_df.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(gen_df.columns)

ax.set_yticklabels(gen_df.columns)

plt.show()
def redundant_pairs(df):

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def top_correlations(df, method, n):

    au_corr = df.corr(method = method).abs().unstack()

    labels_to_drop = redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



top_corr = top_correlations(gen_df,'spearman',57)

top_corr

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from kmodes.kmodes import KModes

per = np.array(gen_df)

km = KModes(n_clusters=500,max_iter=1000,init='Huang',n_init=2,

            n_jobs=-1)



m1 = km.fit(per)

m1.cluster_centroids_
m1.cost_
km = KModes(n_clusters=6,max_iter=1000,init='Huang',n_init=2,

            n_jobs=-1)



m0 = km.fit(per)

m0.cluster_centroids_

m0.cost_
mdl1 = m1.cluster_centroids_

km1 = KModes(n_clusters=250,max_iter=1000,init='Huang',n_init=2,

            n_jobs=-1)



m2 = km1.fit(mdl1)

m2.cost_
mdl2 = m2.cluster_centroids_

km2 = KModes(n_clusters=125,max_iter=1000,init='Huang',n_init=2,

            n_jobs=-1)



m3= km2.fit(mdl2)

m3.cost_
mdl3 = m3.cluster_centroids_

km3 = KModes(n_clusters=62,max_iter=1000,init='Huang',n_init=2,

            n_jobs=-1)



m4= km3.fit(mdl3)

m4.cost_
mdl4 = m4.cluster_centroids_

km4 = KModes(n_clusters=31,max_iter=1000,init='Huang',n_init=2,

            n_jobs=-1)



m5 = km4.fit(mdl4)

m5.cost_
mdl4 = m4.cluster_centroids_

km4 = KModes(n_clusters=31,max_iter=1000,init='Cao',n_init=2,

            n_jobs=-1)



m5 = km4.fit(mdl4)

m5.cost_
mdl5 = m5.cluster_centroids_

km5 = KModes(n_clusters=15,max_iter=1000,init='Cao',n_init=2,

            n_jobs=-1)



m6 = km5.fit(mdl5)

m6.cost_
mdl5 = m5.cluster_centroids_

km5 = KModes(n_clusters=15,max_iter=1000,init='Huang',n_init=2,

            n_jobs=-1)



m6 = km5.fit(mdl5)

m6.cost_
mdl6 = m6.cluster_centroids_

km6 = KModes(n_clusters=7,max_iter=1000,init='Cao',n_init=2,

            n_jobs=-1)



m7 = km6.fit(mdl6)

m7.cost_
mdl7 = m7.cluster_centroids_

km7 = KModes(n_clusters=5,max_iter=1000,init='Cao',n_init=2,

            n_jobs=-1)



m8 = km7.fit(mdl7)

m8.cost_
mdl8 = m8.cluster_centroids_

km8 = KModes(n_clusters=4,max_iter=1000,init='Cao',n_init=2,

            n_jobs=-1)



m9 = km8.fit(mdl8)

m9.cost_
mdl9 = m9.cluster_centroids_

km9 = KModes(n_clusters=3,max_iter=1000,init='Cao',n_init=2,

            n_jobs=-1)



m10 = km9.fit(mdl9)

m10.cost_
mdl9 = m9.cluster_centroids_

mdl9