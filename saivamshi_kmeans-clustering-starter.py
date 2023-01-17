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
df = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

df.head()
df.columns=["ID","Gender","Age","Income","Score"]

df.head()
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

sns.swarmplot(x="Gender",y="Age",data=df)

plt.show()
sns.swarmplot(x="Gender",y="Income",data=df)

plt.show()
fig, axs = plt.subplots(ncols=2,figsize=(12,4))

sns.scatterplot(x="Age",y="Income",data=df,hue="Gender",ax=axs[0])

sns.scatterplot(x="Age",y="Score",data=df,hue="Gender",ax=axs[1])

plt.tight_layout()

plt.show()
sns.relplot("Income","Score",data=df,hue="Gender")
#Dividing into train and test

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.25, random_state=1234)

df_train = df_train.drop(["ID"],axis=1)

df_test = df_test.drop(["ID"],axis=1)

df_train.shape
df_train_2col = df_train[["Income","Score"]]

df_test_2col = df_test[["Income","Score"]]

tmp_train = df_train_2col.copy()
from sklearn.cluster import KMeans

model = KMeans(n_clusters=4)

model.fit(df_train_2col)

df_train_2col["labels"] = model.predict(df_train_2col).copy()

df_test_2col["labels"] = model.predict(df_test_2col).copy()

df_train_2col.head()
fig, axs = plt.subplots(ncols=2,figsize=(12,4))

axs[0].scatter(df_train_2col["Income"],df_train_2col["Score"],c=df_train_2col["labels"],alpha=0.5)

axs[0].set_title("Train Data")

axs[1].scatter(df_test_2col["Income"],df_test_2col["Score"],c=df_test_2col["labels"],alpha=0.5)

axs[1].set_title("Test Data")

centroids_x = model.cluster_centers_[:,0]

centroids_y = model.cluster_centers_[:,1]

axs[0].scatter(centroids_x,centroids_y,marker="D",s=50,c="black")

axs[1].scatter(centroids_x,centroids_y,marker="D",s=50,c="black")
model.inertia_
#Inertia vs Number of Clusters

inertia = []

clus_size=range(1,11)

for k in clus_size:

    model = KMeans(n_clusters=k)

    model.fit(tmp_train)

    inertia.append(model.inertia_)

    

# Plot ks vs inertias

plt.plot(clus_size, inertia, '-o')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.xticks(clus_size)

plt.show()
df_train = pd.get_dummies(df_train,columns=["Gender"],drop_first=True)

df_test = pd.get_dummies(df_test,columns=["Gender"],drop_first=True)

model = KMeans(n_clusters=5)

model.fit(df_train)

labels_train = model.predict(df_train).copy()

labels_test = model.predict(df_test).copy()

res_train.head()
model.inertia_
#Inertia vs Number of Clusters

inertia = []

clus_size=range(1,11)

for k in clus_size:

    model = KMeans(n_clusters=k)

    model.fit(df_train)

    inertia.append(model.inertia_)

    

# Plot ks vs inertias

plt.plot(clus_size, inertia, '-o')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.xticks(clus_size)

plt.show()
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

kmeans = KMeans(n_clusters=5)

pipeline = make_pipeline(scaler,kmeans)

pipeline.fit(df_train)

labels_train_std = pipeline.predict(df_train)

labels_test_std = pipeline.predict(df_test)
labels_train_std