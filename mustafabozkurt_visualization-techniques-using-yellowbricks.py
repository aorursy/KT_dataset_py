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
import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

warnings.filterwarnings("ignore", category=FutureWarning)
df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv", index_col = 0)

df_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv", index_col = 0)

df.head()

df.isnull().sum()
df["Province/State"]=df["Province/State"].fillna(df["Country/Region"])

df.isnull().sum()
df.info
df.describe().T
df.head()
df = df[df["Date"] == max(df["Date"])]

df=df.reset_index(drop=True)

df=df.reset_index(drop=True)





df
df.hist(figsize = (10,10));
df.head()
df=df.drop(["Date","Province/State","Lat","Long"], axis=1)
df=df.groupby(["Country/Region"]).sum()

df=df.reset_index()

df
keys=df["Country/Region"]

keys
df.set_index(keys, drop=False,inplace=True)

df=df.drop(["Country/Region"], axis=1)

df
kmeans = KMeans(n_clusters = 4)
kmeans
k_fit = kmeans.fit(df)
k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_
k_means = KMeans(n_clusters = 2).fit(df)
kumeler = k_means.labels_
kumeler
plt.scatter(df.iloc[:,0], df.iloc[:,1], c = kumeler, s = 50, cmap = "viridis");
merkezler = k_means.cluster_centers_
merkezler
plt.scatter(df.iloc[:,0], df.iloc[:,1], c = kumeler, s = 50, cmap = "viridis")

plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha=0.5);
ssd = []



K = range(1,30)



for k in K:

    kmeans = KMeans(n_clusters = k).fit(df)

    ssd.append(kmeans.inertia_)
plt.plot(K, ssd, "bx-")

plt.xlabel("Total K Distance Against Different K Values")

plt.title("Elbow Method for Optimum Cluster Number")
!pip install yellowbrick
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()

visu = KElbowVisualizer(kmeans, k = (2,20))

visu.fit(df)

visu.poof()
kmeans = KMeans(n_clusters = 4).fit(df)

kmeans
df
kumeler = kmeans.labels_
pd.DataFrame({"Country/Region": df.index, "Kumeler": kumeler})
df["Kume_No"] = kumeler
df.head()
df.sort_values("ConfirmedCases",ascending=False)
df.groupby(["Kume_No"]).sum()
df[df["Kume_No"]==1]
df[df["Kume_No"]==2]
df[df["Kume_No"]==3]
df[df["Kume_No"]==0].head(90)