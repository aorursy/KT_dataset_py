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
# for basic operations

import numpy as np

import pandas as pd

import pandas_profiling



# for data visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# for advanced visualizations 

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected = True)





# for providing path

import os

print(os.listdir('../input/'))



# for model explanation

import shap



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/Iris.csv") 
df.head()
df.describe().T
df.info()
profile=pandas_profiling.ProfileReport(df)

profile
p=df.hist(figsize=(8,8))



g=sns.catplot(x="Species", y="PetalLengthCm", data=df,

             height=5,kind="bar",palette="muted")
g=sns.catplot(x="Species", y="SepalWidthCm", data=df,

             height=5,kind="bar",palette="muted")
g=sns.catplot(x="Species", y="SepalLengthCm", data=df,

             height=5,kind="bar",palette="muted")
g=sns.catplot(x="Species", y="PetalWidthCm", data=df,

             height=5,kind="bar",palette="muted")
df2=df.drop("Id" ,axis=1  )



df2.info()


sz=(9,9)

fig,ax=plt.subplots(figsize=sz)

sns.boxplot(ax=ax, data=df2, orient="h")

data2=df["SepalWidthCm"].max()

data2
data1=df[df["SepalWidthCm"]==4.4]

data1
data4=df["SepalWidthCm"].min()

data4
data3=df[df["SepalWidthCm"]==2]

data3
p=sns.pairplot(df, hue="Species")
plt.scatter(data.PetalLengthCm,data.PetalWidthCm)

plt.show

data5=df.drop(["Species","Id","SepalWidthCm","SepalLengthCm"], axis=1)

data5
from sklearn.cluster import KMeans

wcss = []



for k  in range (1,15):

    kmeans=KMeans(n_clusters=k)

    kmeans.fit(data5)

    wcss.append(kmeans.inertia_)



plt.plot(range(1,15),wcss)

plt.xlabel("number of k (cluster) value")

plt.ylabel("wcss")

plt.show()
kmeans2=KMeans(n_clusters=3)

clusters=kmeans2.fit_predict(data5)

data5["label"]=clusters





    
data5
plt.scatter(data5.PetalLengthCm[data5.label == 0],data5.PetalWidthCm[data5.label == 0],color="red")

plt.scatter(data5.PetalLengthCm[data5.label == 1],data5.PetalWidthCm[data5.label == 1],color="green")

plt.scatter(data5.PetalLengthCm[data5.label == 2],data5.PetalWidthCm[data5.label == 2],color="orange")

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow")

plt.show
from scipy.cluster.hierarchy import linkage, dendrogram



merg=linkage(data5,method="ward")

dendrogram(merg,leaf_rotation=90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
from sklearn.cluster import AgglomerativeClustering



hiyerartical_cluster = AgglomerativeClustering(n_clusters = 3,affinity= "euclidean",linkage = "ward")

cluster = hiyerartical_cluster.fit_predict(data)



data["label"] = cluster



plt.scatter(data5.PetalLengthCm[data5.label == 0 ],data5.PetalWidthCm[data5.label == 0],color = "red")

plt.scatter(data5.PetalLengthCm[data5.label == 1 ],data5.PetalWidthCm[data5.label == 1],color = "green")

plt.scatter(data5.PetalLengthCm[data5.label == 2 ],data5.PetalWidthCm[data5.label == 2],color = "blue")

plt.show()
