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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv("../input/universities/College_Data",index_col=0)

df.head()
df.info()
df.describe(include="all")
df.corr()

#Here we see the correlation between variables.
plt.figure(figsize=(15,10))

mask = np.zeros_like(df.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(df.corr(),cmap="magma",annot=True,linewidths=0, linecolor='white',cbar=True,mask=mask)
plt.figure(figsize=(15,10))

mask = np.zeros_like(df.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(df[df["Private"]== "Yes"].corr(),cmap="plasma",annot=True,linewidths=0, linecolor='white',cbar=True,mask=mask)
plt.figure(figsize=(15,10))

mask = np.zeros_like(df.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(df[df["Private"]== "No"].corr(),cmap="inferno",annot=True,linewidths=0, linecolor='white',cbar=True,mask=mask)
plt.figure(figsize=(15,10))

sns.scatterplot(x="Grad.Rate", y="Room.Board",data=df,hue="Private")
sns.lmplot(x="Outstate", y="F.Undergrad",data=df,hue="Private",size=15)
plt.figure(figsize=(15,10))

df[df["Private"] == "Yes"]["Grad.Rate"].hist(color="blue",bins=50,label="Private = Yes",alpha=0.4)

df[df["Private"] == "No"]["Grad.Rate"].hist(color="red",bins=50,label="Private = No",alpha=0.4)

plt.legend()
g = sns.FacetGrid(df, hue="Private",size=6,aspect=2,palette="husl")

g = g.map(plt.hist, "Grad.Rate",bins=20,alpha=0.6)

sns.set()

plt.legend()
df[df["Grad.Rate"] > 100]
df["Grad.Rate"]["Cazenovia College"] =100
g = sns.FacetGrid(df, hue="Private",size=6,aspect=2,palette="Set2")

g = g.map(plt.hist, "Grad.Rate",bins=20,alpha=0.6)

sns.set()
from IPython.display import Image

url="https://i.stack.imgur.com/FQhxk.jpg"

Image(url,width=800, height=800)
url="https://i.stack.imgur.com/vc01j.png"

Image(url,width=800, height=800)
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2)

#here we reate an instance of a K Means model with 2 clusters.
kmeans.fit(df.drop("Private",axis=1))

#we fit the model to all the data except for the Private labelwhich is the target of the data
kmeans.cluster_centers_

# Here are the cluster center vectors
kmeans.labels_

#Here the algorithm separate the variable into two groups without knowing they are private or state universities.
def converter(private):

    if private == "Yes":

        return 1

    else:

        return 0
df["Cluster"]=df["Private"].apply(converter)

# here we apply our function and create a new column with numerical values as 0's and 1's.
df.head()

#Now our target variable is ready for comparison 
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(df["Cluster"],kmeans.labels_))

print(4*"\n")

print(classification_report(df["Cluster"],kmeans.labels_))