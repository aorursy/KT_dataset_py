# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/para.csv",sep = ";")
df.head()
df.fillna(df.mean(), inplace=True)
fig, ax = plt.subplots(2,2,figsize=(20,10))

k=0

ax[0][0].scatter(df["Energy-1"],df["Amplitude-1"])

ax[0][0].set_xlabel("Energy-1")

ax[0][0].set_ylabel("Amplitude-1")

ax[0][0].set_title("Energy-1 vs Amplitude-1")

ax[0][1].scatter(df["Energy-2"],df["Amplitude-2"])

ax[0][1].set_xlabel("Energy-2")

ax[0][1].set_ylabel("Amplitude-2")

ax[0][1].set_title("Energy-2 vs Amplitude-2")

ax[1][0].scatter(df["RA"],df["Av-Frequency"])

ax[1][0].set_xlabel("RA")

ax[1][0].set_ylabel("Av-Frequency")

ax[1][0].set_title("RA vs Av-Frequency")

ax[1][1].scatter(df["RA"],df["Av-Frequency"])

ax[1][1].set_xlabel("RA")

ax[1][1].set_ylabel("Av-Frequency")

ax[1][1].set_title("RA-2 vs Av-Frequency")

from sklearn.cluster import KMeans

df1 = df[["Energy-1","Amplitude-1"]]

df2 = df[["Energy-2","Amplitude-2"]]

df3 = df[["RA","Av-Frequency"]]

df4 = df[["RA-2","Av-Frequency"]]

total_df = [df1,df2,df3,df4]

l = 0



fig, ax = plt.subplots(2,2,figsize=(20,10))

for i in total_df:

    wcss = []

    for k in range(1,15):

        kmeans = KMeans(n_clusters=k)

        kmeans.fit(i)

        wcss.append(kmeans.inertia_)

    if l == 0:

        ax[l][l].set_title("Energy-1 vs Amplitude-1")

        ax[l][l].plot(range(1,15),wcss)

        ax[l][l].set_xlabel("Number of k (cluster) value")

        ax[l][l].set_ylabel("wcss")

    elif l == 1:

        ax[l-1][l].set_title("Energy-2 vs Amplitude-2")

        ax[l-1][l].plot(range(1,15),wcss)

        ax[l-1][l].set_xlabel("Number of k (cluster) value")

        ax[l-1][l].set_ylabel("wcss")

    elif l == 2:

        ax[l-1][l-2].set_title("RA AV-Frequency")

        ax[l-1][l-2].plot(range(1,15),wcss)

        ax[l-1][l-2].set_xlabel("Number of k (cluster) value")

        ax[l-1][l-2].set_ylabel("wcss")

    elif l ==3:

        ax[l-2][l-2].set_title("RA-2 AV-Frequency")

        ax[l-2][l-2].plot(range(1,15),wcss)

        ax[l-2][l-2].set_xlabel("Number of k (cluster) value")

        ax[l-2][l-2].set_ylabel("wcss")

            

    l += 1

plt.show()
from sklearn.cluster import KMeans

df1.column = ["energy","Amp"]

kmeans2 = KMeans(n_clusters=2,init="random",)

clusters = kmeans2.fit_predict(df1)

df1["label"] = clusters

plt.figure(figsize=(10,5))



plt.scatter(df1["Energy-1"][df1.label ==0],df1["Amplitude-1"][df1.label==0],color="red")

plt.scatter(df1["Energy-1"][df1.label ==1],df1["Amplitude-1"][df1.label==1],color="blue")

plt.ylim(0,150)

plt.xlim(0,20000)

plt.show()
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2,covariance_type="diag").fit(df1)

labels = gmm.predict(df1)

df1["label"] = labels

plt.figure(figsize=(10,5))

plt.scatter(df1["Energy-1"][df1.label ==0],df1["Amplitude-1"][df1.label==0],color="red")

plt.scatter(df1["Energy-1"][df1.label ==1],df1["Amplitude-1"][df1.label==1],color="blue")

#plt.ylim(0,150)

#plt.xlim(0,20000)

plt.show()
from sklearn.cluster import KMeans

kmeans2 = KMeans(n_clusters=2)

clusters = kmeans2.fit_predict(df2)

df2["label"] = clusters

plt.figure(figsize=(10,5))



plt.scatter(df2["Energy-2"][df2.label ==0],df2["Amplitude-2"][df2.label==0],color="red")

plt.scatter(df2["Energy-2"][df2.label ==1],df2["Amplitude-2"][df2.label==1],color="blue")

#plt.ylim(0,150)

#plt.xlim(0,20000)

plt.show()
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2,covariance_type="diag").fit(df2)

labels = gmm.predict(df2)

df2["label"] = labels

plt.figure(figsize=(10,5))

plt.scatter(df2["Energy-2"][df2.label ==0],df2["Amplitude-2"][df2.label==0],color="red")

plt.scatter(df2["Energy-2"][df2.label ==1],df2["Amplitude-2"][df2.label==1],color="blue")

#plt.ylim(0,150)

#plt.xlim(0,20000)

plt.show()
from sklearn.cluster import KMeans

kmeans2 = KMeans(n_clusters=3)

clusters = kmeans2.fit_predict(df3)

df3["label"] = clusters

plt.figure(figsize=(10,5))



plt.scatter(df3["RA"][df3.label ==0],df3["Av-Frequency"][df3.label==0],color="red")

plt.scatter(df3["RA"][df3.label ==1],df3["Av-Frequency"][df3.label==1],color="blue")

plt.scatter(df3["RA"][df3.label ==2],df3["Av-Frequency"][df3.label==2],color="black")

#plt.ylim(0,150)

#plt.xlim(0,20000)

plt.show()
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4,covariance_type="spherical").fit(df3)

labels = gmm.predict(df3)

df3["label"] = labels

plt.figure(figsize=(20,10))

plt.scatter(df3["RA"][df3.label ==0],df3["Av-Frequency"][df3.label==0],color="red")

plt.scatter(df3["RA"][df3.label ==1],df3["Av-Frequency"][df3.label==1],color="blue")

plt.scatter(df3["RA"][df3.label ==2],df3["Av-Frequency"][df3.label==2],color="black")

plt.scatter(df3["RA"][df3.label ==3],df3["Av-Frequency"][df3.label==3],color="yellow")

#plt.ylim(0,150)

#plt.xlim(0,20000)

plt.show()
from sklearn.cluster import KMeans

kmeans2 = KMeans(n_clusters=3)

clusters = kmeans2.fit_predict(df4)

df4["label"] = clusters

plt.figure(figsize=(10,5))



plt.scatter(df4["RA-2"][df4.label ==0],df4["Av-Frequency"][df4.label==0],color="red")

plt.scatter(df4["RA-2"][df4.label ==1],df4["Av-Frequency"][df4.label==1],color="blue")

plt.scatter(df4["RA-2"][df4.label ==2],df4["Av-Frequency"][df4.label==2],color="black")

#plt.ylim(0,150)

#plt.xlim(0,20000)

plt.show()
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4,covariance_type="spherical").fit(df4)

labels = gmm.predict(df4)

df4["label"] = labels

plt.figure(figsize=(20,10))

plt.scatter(df4["RA-2"][df4.label ==0],df4["Av-Frequency"][df4.label==0],color="red")

plt.scatter(df4["RA-2"][df4.label ==1],df4["Av-Frequency"][df4.label==1],color="blue")

plt.scatter(df4["RA-2"][df4.label ==2],df4["Av-Frequency"][df4.label==2],color="black")

plt.scatter(df4["RA-2"][df4.label ==3],df4["Av-Frequency"][df4.label==3],color="yellow")

#plt.ylim(0,150)

#plt.xlim(0,20000)

plt.show()