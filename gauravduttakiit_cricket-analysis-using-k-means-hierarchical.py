import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import datetime as dt



import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

from math import isnan
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv(r"/kaggle/input/cricket/Cricket.csv",encoding='latin1')

df.head()
df.shape
df_dub = df.copy()

# Checking for duplicates and dropping the entire duplicate row if any

df_dub.drop_duplicates(subset=None, inplace=True)
df_dub.shape
df.shape
df.info()
df.describe()
(df.isnull().sum() * 100 / len(df)).value_counts(ascending=False)
df.isnull().sum().value_counts(ascending=False)
(df.isnull().sum(axis=1) * 100 / len(df)).value_counts(ascending=False)
df.isnull().sum(axis=1).value_counts(ascending=False)
df.head()
df[['Strt','End']] = df.Span.str.split("-",expand=True) 
df[['Strt','End']]=df[['Strt','End']].astype(int)

df['Exp']=df['End']-df['Strt']

df=df.drop(['Strt','End','Span'], axis = 1) 

df.head()
#Match Played

plt.figure(figsize = (30,5))

mat = df[['Player','Mat']].sort_values('Mat', ascending = False)

ax = sns.barplot(x='Player', y='Mat', data= mat)

ax.set(xlabel = '', ylabel= 'Match Played')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

mat_top10 = df[['Player','Mat']].sort_values('Mat', ascending = False).head(10)

ax = sns.barplot(x='Player', y='Mat', data= mat_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Match Played')

plt.xticks(rotation=90)

plt.show()
#Inns

plt.figure(figsize = (30,5))

inns = df[['Player','Inns']].sort_values('Inns', ascending = False)

ax = sns.barplot(x='Player', y='Inns', data= inns)

ax.set(xlabel = '', ylabel= 'Innings Played')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

inns_top10 = df[['Player','Inns']].sort_values('Inns', ascending = False).head(10)

ax = sns.barplot(x='Player', y='Inns', data= inns_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Innings Played')

plt.xticks(rotation=90)

plt.show()
#NO

plt.figure(figsize = (30,5))

no = df[['Player','NO']].sort_values('NO', ascending = False)

ax = sns.barplot(x='Player', y='NO', data= no)

ax.set(xlabel = '', ylabel= 'Not Out')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

inns_top10 = df[['Player','NO']].sort_values('NO', ascending = False).head(10)

ax = sns.barplot(x='Player', y='NO', data= inns_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Not Out')

plt.xticks(rotation=90)

plt.show()
#Runs

plt.figure(figsize = (30,5))

run = df[['Player','Runs']].sort_values('Runs', ascending = False)

ax = sns.barplot(x='Player', y='Runs', data= run)

ax.set(xlabel = '', ylabel= 'Runs Scored')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

runs_top10 = df[['Player','Runs']].sort_values('Runs', ascending = False).head(10)

ax = sns.barplot(x='Player', y='Runs', data= runs_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Runs Scored')

plt.xticks(rotation=90)

plt.show()
#HS

df.HS=df.HS.str.extract('(\d+)')

df.HS=df.HS.astype(int)

plt.figure(figsize = (30,5))

hs = df[['Player','HS']].sort_values('HS', ascending = False)

ax = sns.barplot(x='Player', y='HS', data= hs)

ax.set(xlabel = '', ylabel= 'Highest Score')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

hs_top10 = df[['Player','HS']].sort_values('HS', ascending = False).head(10)

ax = sns.barplot(x='Player', y='HS', data= hs_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Highest Score')

plt.xticks(rotation=90)

plt.show()
#Ave

plt.figure(figsize = (30,5))

ave = df[['Player','Ave']].sort_values('Ave', ascending = False)

ax = sns.barplot(x='Player', y='Ave', data= ave)

ax.set(xlabel = '', ylabel= 'Averages')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

avg_top10 = df[['Player','Ave']].sort_values('Ave', ascending = False).head(10)

ax = sns.barplot(x='Player', y='Ave', data= avg_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Averages')

plt.xticks(rotation=90)

plt.show()
#BF

plt.figure(figsize = (30,5))

bf = df[['Player','BF']].sort_values('BF', ascending = False)

ax = sns.barplot(x='Player', y='BF', data= bf)

ax.set(xlabel = '', ylabel= 'Best Form')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

bf_top10 = df[['Player','BF']].sort_values('BF', ascending = False).head(10)

ax = sns.barplot(x='Player', y='BF', data= bf_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Best Form')

plt.xticks(rotation=90)

plt.show()
#SR 

plt.figure(figsize = (30,5))

sr = df[['Player','SR']].sort_values('SR', ascending = False)

ax = sns.barplot(x='Player', y='SR', data= sr)

ax.set(xlabel = '', ylabel= 'SR')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

sr_top10 = df[['Player','SR']].sort_values('SR', ascending = False).head(10)

ax = sns.barplot(x='Player', y='SR', data= sr_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'SR')

plt.xticks(rotation=90)

plt.show()
#100

plt.figure(figsize = (30,5))

r100 = df[['Player','100']].sort_values('100', ascending = False)

ax = sns.barplot(x='Player', y='100', data= r100)

ax.set(xlabel = '', ylabel= "100's Scored" )

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

r100_top10 = df[['Player','100']].sort_values('100', ascending = False).head(10)

ax = sns.barplot(x='Player', y='100', data= r100_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= "100's Scored")

plt.xticks(rotation=90)

plt.show()
#50

plt.figure(figsize = (30,5))

r50 = df[['Player','50']].sort_values('50', ascending = False)

ax = sns.barplot(x='Player', y='50', data= r50)

ax.set(xlabel = '', ylabel= "50s Scored")

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

r50_top10 = df[['Player','50']].sort_values('50', ascending = False).head(10)

ax = sns.barplot(x='Player', y='50', data= r50_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= "50's Scored")

plt.xticks(rotation=90)

plt.show()
#0

plt.figure(figsize = (30,5))

r0 = df[['Player','0']].sort_values('0', ascending = False)

ax = sns.barplot(x='Player', y='0', data= r0)

ax.set(xlabel = '', ylabel= "Os Scored")

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

r0_top10 = df[['Player','0']].sort_values('0', ascending = False).head(10)

ax = sns.barplot(x='Player', y='0', data= r0_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= "Os Scored")

plt.xticks(rotation=90)

plt.show()
#Exp

plt.figure(figsize = (30,5))

exp = df[['Player','Exp']].sort_values('Exp', ascending = False)

ax = sns.barplot(x='Player', y='Exp', data= exp)

ax.set(xlabel = '', ylabel= 'Experience')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

exp_top10 = df[['Player','Exp']].sort_values('Exp', ascending = False).head(10)

ax = sns.barplot(x='Player', y='Exp', data= exp_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Exprience')

plt.xticks(rotation=90)

plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (10, 10))

sns.heatmap(df.corr(), annot = True, cmap="rainbow")

plt.savefig('Correlation')

plt.show()
sns.pairplot(df,corner=True,diag_kind="kde")

plt.show()
df.describe()
f, axes = plt.subplots(4,3, figsize=(16, 8))

s=sns.violinplot(y=df.Exp,ax=axes[0, 0])

axes[0, 0].set_title('Exp')

s=sns.violinplot(y=df.Mat,ax=axes[0, 1])

axes[0, 1].set_title('Mat')

s=sns.violinplot(y=df.Inns,ax=axes[0, 2])

axes[0, 2].set_title('Inns')



s=sns.violinplot(y=df.NO,ax=axes[1, 0])

axes[1, 0].set_title('NO')

s=sns.violinplot(y=df.Runs,ax=axes[1, 1])

axes[1, 1].set_title('Runs')

s=sns.violinplot(y=df.HS,ax=axes[1, 2])

axes[1, 2].set_title('HS')



s=sns.violinplot(y=df.Ave,ax=axes[2, 0])

axes[2, 0].set_title('Ave')

s=sns.violinplot(y=df.SR,ax=axes[2, 1])

axes[2, 1].set_title('SR')

s=sns.violinplot(y=df['100'],ax=axes[2, 2])

axes[2, 2].set_title('100')

s=sns.violinplot(y=df.BF,ax=axes[3, 0])

axes[3, 0].set_title('BF')

s=sns.violinplot(y=df['50'],ax=axes[3, 1])

axes[3, 1].set_title('50s')

s=sns.violinplot(y=df['0'],ax=axes[3, 2])

axes[3, 2].set_title('0s')

plt.show()
plt.figure(figsize = (30,10))

features=[ 'Mat', 'Inns', 'NO', 'Runs', 'HS', 'Ave', 'BF', 'SR', '100','50', '0', 'Exp']

for i in enumerate(features):

    plt.subplot(3,4,i[0]+1)

    sns.distplot(df[i[1]])
Q3 = df.Mat.quantile(0.99)

Q1 = df.Mat.quantile(0.01)

df['Mat'][df['Mat']<=Q1]=Q1

df['Mat'][df['Mat']>=Q3]=Q3
Q3 = df.Inns.quantile(0.99)

Q1 = df.Inns.quantile(0.01)

df['Inns'][df['Inns']<=Q1]=Q1

df['Inns'][df['Inns']>=Q3]=Q3
Q3 = df.NO.quantile(0.99)

Q1 = df.NO.quantile(0.01)

df['NO'][df['NO']<=Q1]=Q1

df['NO'][df['NO']>=Q3]=Q3
Q3 = df.Runs.quantile(0.99)

Q1 = df.Runs.quantile(0.01)

df['Runs'][df['Runs']<=Q1]=Q1

df['Runs'][df['Runs']>=Q3]=Q3
Q3 = df.HS.quantile(0.99)

Q1 = df.HS.quantile(0.01)

df['HS'][df['HS']<=Q1]=Q1

df['HS'][df['HS']>=Q3]=Q3
Q3 = df.Ave.quantile(0.99)

Q1 = df.Ave.quantile(0.01)

df['Ave'][df['Ave']<=Q1]=Q1

df['Ave'][df['Ave']>=Q3]=Q3
Q3 = df.BF.quantile(0.99)

Q1 = df.BF.quantile(0.01)

df['BF'][df['BF']<=Q1]=Q1

df['BF'][df['BF']>=Q3]=Q3
Q3 = df.SR.quantile(0.99)

Q1 = df.SR.quantile(0.01)

df['SR'][df['SR']<=Q1]=Q1

df['SR'][df['SR']>=Q3]=Q3
Q3 = df.Exp.quantile(0.99)

Q1 = df.Exp.quantile(0.01)

df['Exp'][df['Exp']<=Q1]=Q1

df['Exp'][df['Exp']>=Q3]=Q3
Q3 = df['100'].quantile(0.99)

Q1 = df['100'].quantile(0.01)

df['100'][df['100']<=Q1]=Q1

df['100'][df['100']>=Q3]=Q3
Q3 = df['50'].quantile(0.99)

Q1 = df['50'].quantile(0.01)

df['50'][df['50']<=Q1]=Q1

df['50'][df['50']>=Q3]=Q3
Q3 = df['0'].quantile(0.99)

Q1 = df['0'].quantile(0.01)

df['0'][df['0']<=Q1]=Q1

df['0'][df['0']>=Q3]=Q3
f, axes = plt.subplots(4,3, figsize=(16, 8))

s=sns.violinplot(y=df.Exp,ax=axes[0, 0])

axes[0, 0].set_title('Exp')

s=sns.violinplot(y=df.Mat,ax=axes[0, 1])

axes[0, 1].set_title('Mat')

s=sns.violinplot(y=df.Inns,ax=axes[0, 2])

axes[0, 2].set_title('Inns')



s=sns.violinplot(y=df.NO,ax=axes[1, 0])

axes[1, 0].set_title('NO')

s=sns.violinplot(y=df.Runs,ax=axes[1, 1])

axes[1, 1].set_title('Runs')

s=sns.violinplot(y=df.HS,ax=axes[1, 2])

axes[1, 2].set_title('HS')



s=sns.violinplot(y=df.Ave,ax=axes[2, 0])

axes[2, 0].set_title('Ave')

s=sns.violinplot(y=df.SR,ax=axes[2, 1])

axes[2, 1].set_title('SR')

s=sns.violinplot(y=df['100'],ax=axes[2, 2])

axes[2, 2].set_title('100')

s=sns.violinplot(y=df.BF,ax=axes[3, 0])

axes[3, 0].set_title('BF')

s=sns.violinplot(y=df['50'],ax=axes[3, 1])

axes[3, 1].set_title('50s')

s=sns.violinplot(y=df['0'],ax=axes[3, 2])

axes[3, 2].set_title('0s')

plt.show()
# Dropping Player field as final dataframe will only contain data columns



df_drop = df.copy()

player = df_drop.pop('Player')
df_drop.head()
# Calculating Hopkins score to know whether the data is good for clustering or not.



def hopkins(X):

    d = X.shape[1]

    n = len(X)

    m = int(0.1 * n) 

    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

 

    rand_X = sample(range(0, n, 1), m)

 

    ujd = []

    wjd = []

    for j in range(0, m):

        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)

        ujd.append(u_dist[0][1])

        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)

        wjd.append(w_dist[0][1])

 

    HS = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(HS):

        print(ujd, wjd)

        HS = 0

 

    return HS
# Hopkins score

Hopkins_score=round(hopkins(df_drop),2)
print("{} is a good Hopkins score for Clustering.".format(Hopkins_score))


scaler = StandardScaler()

df_scaled = scaler.fit_transform(df_drop)

df_scaled 
df_df1 = pd.DataFrame(df_scaled, columns = [ 'Mat', 'Inns', 'NO', 'Runs', 'HS', 'Ave', 'BF', 'SR', '100',

                                            '50', '0', 'Exp'])

df_df1.head()
# Elbow curve method to find the ideal number of clusters.

clusters=list(range(2,8))

ssd = []

for num_clusters in clusters:

    model_clus = KMeans(n_clusters = num_clusters, max_iter=150,random_state= 50)

    model_clus.fit(df_df1)

    ssd.append(model_clus.inertia_)



plt.plot(clusters,ssd);
# Silhouette score analysis to find the ideal number of clusters for K-means clustering



range_n_clusters = [2, 3, 4, 5, 6, 7, 8]



for num_clusters in range_n_clusters:

    

    # intialise kmeans

    kmeans = KMeans(n_clusters=num_clusters, max_iter=50,random_state= 100)

    kmeans.fit(df_df1)

    

    cluster_labels = kmeans.labels_

    

    # silhouette score

    silhouette_avg = silhouette_score(df_df1, cluster_labels)

    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
#K-means with k=4 clusters



cluster = KMeans(n_clusters=4, max_iter=150, random_state= 15)

cluster.fit(df_df1)
# Cluster labels



cluster.labels_
# Assign the label



df['Cluster_Id'] = cluster.labels_

df.head()
## Number of countries in each cluster

df.Cluster_Id.value_counts(ascending=True)
# Scatter plot on Original attributes to visualize the spread of the data



plt.figure(figsize = (20,15))

plt.subplot(3,1,1)

sns.scatterplot(x = 'Ave', y = 'NO',hue='Cluster_Id',data = df,legend='full',palette="Set1")

plt.subplot(3,1,2)

sns.scatterplot(x = 'Ave', y = 'SR',hue='Cluster_Id', data = df,legend='full',palette="Set1")

plt.subplot(3,1,3)

sns.scatterplot(x = 'NO', y = 'SR',hue='Cluster_Id', data=df,legend='full',palette="Set1")

plt.show()

 #Violin plot on Original attributes to visualize the spread of the data



fig, axes = plt.subplots(2,2, figsize=(15,12))



sns.violinplot(x = 'Cluster_Id', y = 'Ave', data = df,ax=axes[0][0])

sns.violinplot(x = 'Cluster_Id', y = 'SR', data = df,ax=axes[0][1])

sns.violinplot(x = 'Cluster_Id', y = 'NO', data=df,ax=axes[1][0])

sns.violinplot(x = 'Cluster_Id', y = 'Exp', data=df,ax=axes[1][1])

plt.show()
df[['NO','Ave','SR','Cluster_Id']].groupby('Cluster_Id').mean()
ax=df[['NO','Ave','SR','Cluster_Id']].groupby('Cluster_Id').mean().plot(kind = 'bar',figsize = (15,5))



for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=0)

plt.show();
df[df['Cluster_Id']==0].sort_values(by = ['NO','Ave','SR'], ascending = [True,False,False]).head()

#Can be Batsman Coach
df[df['Cluster_Id']==1].sort_values(by = ['NO','Ave','SR'], ascending = [True,False,False]).head()

#Openers
df[df['Cluster_Id']==2].sort_values(by = ['NO','Ave','SR'], ascending = [True,False,False]).head()

#Chockers 
df[df['Cluster_Id']==3].sort_values(by = ['NO','Ave','SR'], ascending = [True,False,False]).head()

#Hitters
FinalListbyKMean=df[df['Cluster_Id']==3].sort_values(by = ['NO','Ave','SR'], ascending = [True,False,False])

FinalListbyKMean['Player']

FinalListbyKMean.reset_index(drop=True).Player[:]
df_list_no = pd.DataFrame(FinalListbyKMean.groupby(['Player'])['NO'].mean().sort_values(ascending = True)).head()

ax=df_list_no.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.title('Player & Not Out')

plt.xlabel("Player",fontweight = 'bold')

plt.ylabel("Not Out", fontsize = 12, fontweight = 'bold')

plt.show()
df_list_ave = pd.DataFrame(FinalListbyKMean.groupby(['Player'])['Ave'].mean().sort_values(ascending = False)).head()

ax=df_list_ave.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.title('Player & Averages')

plt.xlabel("Player",fontweight = 'bold')

plt.ylabel("Averages", fontsize = 12, fontweight = 'bold')

plt.show()
df_list_sr = pd.DataFrame(FinalListbyKMean.groupby(['Player'])['SR'].mean().sort_values(ascending = False)).head()

ax=df_list_sr.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.title('Player & Strike Rates')

plt.xlabel("Player",fontweight = 'bold')

plt.ylabel("Strike Rates", fontsize = 12, fontweight = 'bold')

plt.show()
df_df1.head()

# Single linkage

plt.figure(figsize = (20,10))

mergings = linkage(df_df1, method='single',metric='euclidean')

dendrogram(mergings)

plt.show()
# Complete Linkage

plt.figure(figsize = (20,10))

mergings = linkage(df_df1, method='complete',metric='euclidean')

dendrogram(mergings)

plt.show()
df_hc = df.copy()

df_hc = df_hc.drop('Cluster_Id',axis=1)

df_hc.head()
# 4 clusters

cluster_labels = cut_tree(mergings, n_clusters=4).reshape(-1, )

cluster_labels
# assign cluster labels

df_hc['Cluster_labels'] = cluster_labels

df_hc.head()
## Number of countries in each cluster

df_hc.Cluster_labels.value_counts(ascending=True)
# Scatter plot on Original attributes to visualize the spread of the data



plt.figure(figsize = (20,15))

plt.subplot(3,1,1)

sns.scatterplot(x = 'Ave', y = 'NO',hue='Cluster_labels',data = df_hc,legend='full',palette="Set1")

plt.subplot(3,1,2)

sns.scatterplot(x = 'Ave', y = 'SR',hue='Cluster_labels', data = df_hc,legend='full',palette="Set1")

plt.subplot(3,1,3)

sns.scatterplot(x = 'NO', y = 'SR',hue='Cluster_labels', data=df_hc,legend='full',palette="Set1")

plt.show()

 #Violin plot on Original attributes to visualize the spread of the data



fig, axes = plt.subplots(2,2, figsize=(15,12))



sns.violinplot(x = 'Cluster_labels', y = 'Ave', data = df_hc,ax=axes[0][0])

sns.violinplot(x = 'Cluster_labels', y = 'SR', data = df_hc,ax=axes[0][1])

sns.violinplot(x = 'Cluster_labels', y = 'NO', data=df_hc,ax=axes[1][0])

sns.violinplot(x = 'Cluster_labels', y = 'Exp', data=df_hc,ax=axes[1][1])

plt.show()
df_hc[['NO','Ave','SR','Cluster_labels']].groupby('Cluster_labels').mean()
ax=df_hc[['NO','Ave','SR','Cluster_labels']].groupby('Cluster_labels').mean().plot(kind = 'bar',figsize = (15,5))



for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation=0)

plt.show();
df_hc[df_hc['Cluster_labels']==0].sort_values(by = ['NO','Ave','SR'], ascending = [True,False,False]).head()

#Can be Batsman Coach
df_hc[df_hc['Cluster_labels']==1].sort_values(by = ['NO','Ave','SR'], ascending = [True,False,False]).head()

#MiddleOrders 
df_hc[df_hc['Cluster_labels']==2].sort_values(by = ['NO','Ave','SR'], ascending = [True,False,False]).head()

#Hitters
df_hc[df_hc['Cluster_labels']==3].sort_values(by = ['NO','Ave','SR'], ascending = [True,False,False]).head()

#Hitters 2 
FinalListbyHC=df_hc[df_hc['Cluster_labels']==2].sort_values(by = ['NO','Ave','SR'], ascending = [True,False,False])

FinalListbyHC['Player']

FinalListbyHC.reset_index(drop=True).Player[:]
df_list_hc = pd.DataFrame(FinalListbyHC.groupby(['Player'])['NO'].mean().sort_values(ascending = True)).head()

ax=df_list_no.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.title('Player & Not Out')

plt.xlabel("Player",fontweight = 'bold')

plt.ylabel("Not Out", fontsize = 12, fontweight = 'bold')

plt.show()
df_list_ave = pd.DataFrame(FinalListbyHC.groupby(['Player'])['Ave'].mean().sort_values(ascending = False)).head()

ax=df_list_ave.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.title('Player & Averages')

plt.xlabel("Player",fontweight = 'bold')

plt.ylabel("Averages", fontsize = 12, fontweight = 'bold')

plt.show()
df_list_sr = pd.DataFrame(FinalListbyHC.groupby(['Player'])['SR'].mean().sort_values(ascending = False)).head()

ax=df_list_sr.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.title('Player & Strike Rates')

plt.xlabel("Player",fontweight = 'bold')

plt.ylabel("Strike Rates", fontsize = 12, fontweight = 'bold')

plt.show()
## Number of countries in each cluster in K-Means 

df.Cluster_Id.value_counts(ascending=True)
## Number of countries in each cluster for Hierarchical clustering

df_hc.Cluster_labels.value_counts(ascending=True)
FinalListbyKMean=df[df['Cluster_Id']==3].sort_values(by = ['NO','Ave','SR'], ascending = [True,False,False])

FinalListbyKMean['Player']

FinalListbyKMean.reset_index(drop=True).Player[:5]
df_list_no = pd.DataFrame(FinalListbyKMean.groupby(['Player'])['NO'].mean().sort_values(ascending = True)).head()

ax=df_list_no.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.title('Player & Not Out')

plt.xlabel("Player",fontweight = 'bold')

plt.ylabel("Not Out", fontsize = 12, fontweight = 'bold')

plt.show()
# plots

df_list_ave = pd.DataFrame(FinalListbyKMean.groupby(['Player'])['Ave'].mean().sort_values(ascending = False)).head()

ax=df_list_ave.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.title('Player & Averages')

plt.xlabel("Player",fontweight = 'bold')

plt.ylabel("Averages", fontsize = 12, fontweight = 'bold')

plt.show()
df_list_sr = pd.DataFrame(FinalListbyKMean.groupby(['Player'])['SR'].mean().sort_values(ascending = False)).head()

ax=df_list_sr.plot(kind = 'bar',figsize = (10,5))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.title('Player & Strike Rates')

plt.xlabel("Player",fontweight = 'bold')

plt.ylabel("Strike Rates", fontsize = 12, fontweight = 'bold')

plt.show()
#FinalList with K-Means

FinalListbyKMean.reset_index(drop=True).Player[:5]
# Final Players list with Hierarchical clustering

FinalListbyHC.reset_index(drop=True).Player[:5]
# Final Players list

FinalListbyKMean.reset_index(drop=True).Player[:5]