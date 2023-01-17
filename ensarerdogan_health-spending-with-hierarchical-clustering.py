# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AgglomerativeClustering



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')

data.head()
data.index=data.iloc[:,0]  

data.index
data.index
data1=data.drop(["country"],axis=1)

data1.head()
data1.isnull().sum()
data1.describe().T
lnk=linkage(data1,method="ward")

dendrogram(lnk,leaf_rotation=90)

plt.xlabel("data")

plt.ylabel("Noktalar arası uzaklık")

plt.show()
hierarchical_cluster=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")

Y_tahmin=hierarchical_cluster.fit_predict(data1)

print(Y_tahmin)
data1["label"]=Y_tahmin

data1.head()
label0=data1[data1.label==0]

label0.info()
label1=data1[data1.label==1]

label1.info()
label2=data1[data1.label==2]

label2.info()
sns.scatterplot(x="gdpp",y="health",color="red",data=label0)

sns.scatterplot(x="gdpp",y="health",color="blue",data=label1)

sns.scatterplot(x="gdpp",y="health",color="green",data=label2)

plt.show()