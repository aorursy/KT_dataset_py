
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings 
import math
import sys

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_raw = pd.read_csv("../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv", sep="\t")

data_raw.rename(columns={"EXT1":"life of party", "EXT2":"don't talk a lot", "EXT3": "comfortable around people",
                    "EXT4":"keep background", "EXT5":"start conversation", "EXT6":"little to say", "EXT7":"talk to party people",
                     "EXT8":"don't like draw attention", "EXT9":"center of attention", "EXT10":"quite around strangers"}, inplace = True)

#filter all countries with less then 5 values. Sadly this one is reeeaaaaaally slow.
data=pd.DataFrame()
counter=1
last=len(pd.unique(data_raw["country"]))
#only use entries with more then 5 values
for country in pd.unique(data_raw["country"]):
    print('\r', counter, "of", last, end="")
    counter=counter+1
    if len(data_raw[data_raw["country"]==country])>=5:
        data=data.append(data_raw[data_raw["country"]==country])
with warnings.catch_warnings():
    # there are some warnings about runtime, I gonna ignore those.
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for i in range(0,9):
        pyplot.hist(data.iloc[:,i],  alpha=1, label=data.columns[i])
        pyplot.legend(loc='upper right')
        plt.title(data.columns.values[i])
        plt.xlabel("Value of votes")
        plt.ylabel("Number of votes")
        pyplot.show()

correlations = data.iloc[:,1:10].corr()

ax = sns.heatmap(
    correlations, 
    vmin=-1, vmax=1, center=0,
    square=True,
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
ax.set_title("correlation matrix")
df_coun1 = pd.DataFrame()

with warnings.catch_warnings():
    # there are some warnings about runtime, I gonna ignore those.
    warnings.simplefilter("ignore", category=RuntimeWarning)
    
    for coun in pd.unique(data["country"]):
        data_coun = data[data["country"]==coun].iloc[:,0]
        data_coun = data_coun.dropna(0)
        mean_coun = data_coun.values.mean()
        std_coun = data_coun.values.std(ddof=1)
        df_coun1 = df_coun1.append([[coun, mean_coun, std_coun]])

    df_coun1=df_coun1.dropna(0)
    df_coun1.columns=["country code", "mean", "stdev"]

    pyplot.hist(df_coun1.iloc[:,1],  alpha=0.5)

    plt.xlabel("Mean")
    plt.ylabel("Number")
    
    plt.title(data.columns.values[0])
    
    pyplot.figure()
    pyplot.scatter(df_coun1.iloc[:,1], df_coun1.iloc[:,2])
    plt.xlabel("Mean")
    plt.ylabel("Stdev")
    
    plt.title(data.columns.values[0])

    for i in range(len(df_coun1.index)):
        plt.annotate(df_coun1.iloc[i,0], (df_coun1.iloc[i,1], df_coun1.iloc[i,2]))

#I want to cluster the results. 
df_kmeans = df_coun1.iloc[:,1:3]


cluster=KMeans(n_clusters=4).fit(df_kmeans)
centroids = cluster.cluster_centers_
plt.scatter(df_kmeans.iloc[:,0], df_kmeans.iloc[:,1], c= cluster.labels_.astype(float), alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel("Mean")
plt.ylabel("Stdev")
plt.title(data.columns.values[0])


for i in range(len(df_coun1.index)):
    plt.annotate(df_coun1.iloc[i,0], (df_coun1.iloc[i,1], df_coun1.iloc[i,2]))
plt.show()

print (df_coun1[df_coun1.iloc[:,1]>3])
df_coun2 = pd.DataFrame()

with warnings.catch_warnings():
    # there are some warnings about runtime, I gonna ignore those.
    warnings.simplefilter("ignore", category=RuntimeWarning)
    
    for coun in pd.unique(data["country"]):
        data_coun = data[data["country"]==coun].iloc[:,1]
        data_coun = data_coun.dropna(0)
        mean_coun = data_coun.values.mean()
        std_coun = data_coun.values.std(ddof=1)
        df_coun2 = df_coun2.append([[coun, mean_coun, std_coun]])
    print(df_coun2)
    df_coun2=df_coun2.dropna(0)
    df_coun2.columns=["country code", "mean", "stdev"]

    pyplot.hist(df_coun2.iloc[:,1],  alpha=0.5)

    plt.xlabel("Mean")
    plt.ylabel("Number")
    
    plt.title(data.columns.values[1])
    
    pyplot.figure()
    pyplot.scatter(df_coun2.iloc[:,1], df_coun2.iloc[:,2])
    plt.xlabel("Mean")
    plt.ylabel("Stdev")
    
    plt.title(data.columns.values[1])

    for i in range(len(df_coun2.index)):
        plt.annotate(df_coun2.iloc[i,0], (df_coun2.iloc[i,1], df_coun2.iloc[i,2]))
#I want to cluster the results. 
df_kmeans = df_coun2.iloc[:,1:3]


cluster=KMeans(n_clusters=4).fit(df_kmeans)
centroids = cluster.cluster_centers_
plt.scatter(df_kmeans.iloc[:,0], df_kmeans.iloc[:,1], c= cluster.labels_.astype(float), alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel("Mean")
plt.ylabel("Stdev")
plt.title(data.columns.values[1])


for i in range(len(df_coun2.index)):
    plt.annotate(df_coun2.iloc[i,0], (df_coun2.iloc[i,1], df_coun2.iloc[i,2]))
plt.show()

print (df_coun2[df_coun2.iloc[:,1]>3.3])
print("Your country lifes the live of a party:\n", df_coun2[df_coun2["country code"]=="DE"])
print("Your country does not speak a lot around people:\n", df_coun1[df_coun1["country code"]=="DE"] )