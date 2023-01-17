# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/clicks-conversion-tracking/KAG_conversion_data.csv")
df.head(50)
df.shape
list(df.columns)
df.head()

import seaborn as sns
import matplotlib.pyplot as plt
g = sns.FacetGrid(df, col="gender", hue="age")
g.map(plt.scatter, "Impressions", "Clicks", alpha=.7)
g.add_legend();
g = sns.FacetGrid(df, col="gender", hue="age")
g.map(plt.scatter, "Clicks", "Total_Conversion", alpha=.7)
g.add_legend();
g = sns.FacetGrid(df, col="gender", hue="age")
g.map(plt.scatter, "Total_Conversion", "Approved_Conversion", alpha=.7)
g.add_legend();
g = sns.FacetGrid(df, col="gender", hue="age")
g.map(plt.scatter, "interest", "Clicks", alpha=.7)
g.add_legend();
g = sns.FacetGrid(df, col="gender", hue="age")
g.map(plt.scatter, "fb_campaign_id", "Clicks", alpha=.7)
g.add_legend();
g = sns.FacetGrid(df, col="gender", hue="age")
g.map(plt.scatter, "interest", "Approved_Conversion", alpha=.7)
g.add_legend();
g = sns.FacetGrid(df, col="gender", hue="age")
g.map(plt.scatter, "interest", "Spent", alpha=.7)
g.add_legend();
g = sns.FacetGrid(df, hue="gender")
g.map(plt.scatter, "interest", "age","Clicks", alpha=.7)
g.add_legend();
g = sns.FacetGrid(df, col="age", hue="gender")
g.map(plt.scatter, "interest", "Approved_Conversion", alpha=.7)
g.add_legend();
bins = 10
g = sns.FacetGrid(df, col="age",hue="gender", palette="Set1", col_wrap=2)
g.map(plt.hist, 'fb_campaign_id', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()
bins = 10
g = sns.FacetGrid(df, col="age",hue="gender", palette="Set1", col_wrap=2)
g.map(plt.hist, 'xyz_campaign_id',bins=bins, ec="k")

g.axes[-1].legend()
plt.show()
sns.catplot(x="xyz_campaign_id", y="Approved_Conversion",hue="gender", col="age",kind="bar",data=df,col_wrap=2)
sns.catplot(x="xyz_campaign_id", y="Impressions",hue="gender", col="age",kind="bar",data=df,col_wrap=2)
sns.catplot(x="xyz_campaign_id", y="Approved_Conversion",hue="age",kind="bar",data=df)
sns.catplot(x="xyz_campaign_id", y="Spent",hue="gender", col="age",kind="bar",data=df,col_wrap=2)
g = sns.FacetGrid(df, hue="gender", palette="Set1", height=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "xyz_campaign_id", "Clicks", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
g = sns.FacetGrid(df, hue="gender", palette="Set1", height=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "age", "Clicks", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
g = sns.FacetGrid(df, hue="gender", palette="Set1", height=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "Clicks", "Total_Conversion", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
g = sns.FacetGrid(df, hue="gender", palette="Set1", height=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "interest", "Clicks", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
df['gender'].replace(to_replace=['M','F'], value=[0,1],inplace=True)
df.head()
df['age'].replace(to_replace=['30-34', '35-39', '40-44', '45-49'], value=[0,1,2,3],inplace=True)
df.head()

X=df
X = df.values
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler


X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet
clusterNum = 4
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
df["Clus_km"] = labels
df.head(5)





df.groupby('Clus_km').mean()
print(X[:, ])
area = np.pi * ( X[:, 4])**2  
plt.scatter(X[:, 3], X[:, 5], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Interest', fontsize=18)
plt.ylabel('Age', fontsize=16)

plt.show()
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('gender')
ax.set_ylabel('Age')
ax.set_zlabel('Interest')

ax.scatter(X[:, 4], X[:, 3], X[:, 5], c= labels.astype(np.float))

print(X[:, ])
area = np.pi * ( X[:, 3])**2  
plt.scatter(X[:, 4], X[:, 7], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('gender', fontsize=18)
plt.ylabel('Clicks', fontsize=16)

plt.show()
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('age')
ax.set_ylabel('gender')
ax.set_zlabel('Clicks')

ax.scatter(X[:, 3], X[:, 4], X[:, 7], c= labels.astype(np.float))
