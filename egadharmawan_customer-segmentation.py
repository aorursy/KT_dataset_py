import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(rc={'figure.figsize':(10,10)})

sns.set_style("whitegrid")

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/german-credit/german_credit_data.csv', index_col=0)

df.head()
df.info()
df.isnull().sum()
## Age Dist



sns.distplot(df.Age, hist=True, rug=True,color='c')
## Gender Dist



df.Sex.value_counts().plot.bar()
## Customer by age and sex



ax = sns.boxplot(x="Sex", y="Age",

                 data=df, palette="viridis")
## Corr between number of checking account and Credit amount by Sex



ax = sns.violinplot(x="Checking account", y="Credit amount", hue='Sex',

                 data=df, palette="plasma")
## People housing for each Job and their credit amount



ax = sns.boxplot(x="Job", y="Credit amount", hue='Housing',

                 data=df, palette="plasma")
## Corr between number of checking account and Credit amount by Job / Skill



ax = sns.swarmplot(x="Checking account", y="Credit amount", hue='Job',

                 data=df, palette="spring")
## Corr between Credit amount and Purpose by Job



ax = sns.boxplot(x="Purpose", y="Credit amount", hue='Job',

                 data=df, palette="Pastel1")
## Corr between Credit Amount and Age by Sex



ax = sns.scatterplot(x="Credit amount", y="Age", hue='Sex',

                 data=df, palette="rainbow")
## Corr between Credit Amount and Age by Purpose



ax = sns.scatterplot(x="Credit amount", y="Age", hue='Purpose',

                 data=df, palette="jet_r")
import sklearn.preprocessing as pre

from scipy.special import inv_boxcox

from scipy.stats import boxcox
## Categorical Data



df = pd.get_dummies(df)

df.head()
## heatmap



sns.heatmap(df.corr(), cmap='twilight')
# Cluster column



Cluster = df.loc[:,["Age","Credit amount", "Duration"]]
## Scalling



fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))

print('Skew Value : ' + str(Cluster.Age.skew()))

sns.distplot(Cluster["Age"], ax=ax1)

print('Skew Value : ' + str(Cluster['Credit amount'].skew()))

sns.distplot(Cluster["Credit amount"], ax=ax2)

print('Skew Value : ' + str(Cluster.Duration.skew()))

sns.distplot(Cluster["Duration"], ax=ax3)

plt.tight_layout()
def scalling(df, column):

    f = plt.figure(figsize=(15,13))



    # log 1 Transform

    ax = f.add_subplot(221)

    L1p = np.log1p(df[column])

    sns.distplot(L1p,color='b',ax=ax)

    ax.set_title('skew value Log 1 transform: ' + str(np.log1p(df[column]).skew()))



    # Square Log Transform

    ax = f.add_subplot(222)

    SRT = np.sqrt(df[column])

    sns.distplot(SRT,color='c',ax=ax)

    ax.set_title('Skew Value Square Transform: ' + str(np.sqrt(df[column]).skew()))



    # Log Transform

    ax = f.add_subplot(223)

    LT = np.log(df[column])

    sns.distplot(LT, color='r',ax=ax)

    ax.set_title('Skew value Log Transform: ' + str(np.log(df[column]).skew()))



    # Box Cox Transform

    ax = f.add_subplot(224)

    BCT,fitted_lambda = boxcox(df[column],lmbda=None)

    sns.distplot(BCT,color='g',ax=ax)

    ax.set_title('Skew Value Box Cox Transform: ' + str(pd.Series(BCT).skew()))
scalling(Cluster, 'Age')
scalling(Cluster, 'Credit amount')
scalling(Cluster, 'Duration')
## Apply Transformation



Cluster['Age'],fitted_lambda = boxcox(Cluster['Age'],lmbda=None)

Cluster['Credit amount'], fitted_lambda = boxcox(Cluster['Credit amount'],lmbda=None)

Cluster['Duration'], fitted_lambda = boxcox(Cluster['Duration'],lmbda=None)

Cluster.head()
from sklearn.cluster import KMeans 

from sklearn import metrics 

from scipy.spatial.distance import cdist

from mpl_toolkits.mplot3d import Axes3D
distortions = []

mapping1 = {}

K = range(1,15) 



for k in K:

    kmeanModel = KMeans(n_clusters=k).fit(Cluster)

    kmeanModel.fit(Cluster)



    distortions.append(sum(np.min(cdist(Cluster, kmeanModel.cluster_centers_, 

                    'euclidean'),axis=1)) / Cluster.shape[0]) 



    mapping1[k] = sum(np.min(cdist(Cluster, kmeanModel.cluster_centers_, 

                'euclidean'),axis=1)) / Cluster.shape[0] 
for key,val in mapping1.items(): 

    print(str(key)+' : '+str(val)) 
plt.plot(K, distortions, 'bx-') 

plt.xlabel('Values of K') 

plt.ylabel('Distortion') 

plt.title('The Elbow Method using Distortion') 

plt.show() 
kmeans = KMeans(n_clusters = 4)

kmeans.fit(Cluster)

y_pred = kmeans.predict(Cluster)

print(kmeans.cluster_centers_)
df["label"] = kmeans.labels_

df.head()
fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(df["Credit amount"], df["Duration"], df["Age"], c=y_pred, cmap='jet_r')

ax.set_xlabel("Credit amount")

ax.set_ylabel("Duration")

ax.set_zlabel("Age")
## Detailed Overview



f = plt.figure(figsize=(15,13))

ax = f.add_subplot(311)

ax = sns.scatterplot(x="Credit amount", y="Duration", hue='label', data=df, palette="jet_r")

ax = f.add_subplot(312)

ax = sns.scatterplot(x="Age", y="Credit amount", hue='label', data=df, palette="jet_r")

ax = f.add_subplot(313)

ax = sns.scatterplot(x="Age", y="Duration", hue='label', data=df, palette="jet_r")