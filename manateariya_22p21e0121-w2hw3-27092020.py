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
from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram

from sklearn.datasets import load_iris

from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing  import LabelEncoder





data=pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data.columns

data.head(5)
data['neighbourhood']= data[['neighbourhood']].apply( LabelEncoder().fit_transform)['neighbourhood']

data['neighbourhood_group']= data[['neighbourhood_group']].apply( LabelEncoder().fit_transform)['neighbourhood_group']



data.head(5)
data.columns
data1=data[['neighbourhood','price']]
data1
from sklearn.cluster import KMeans

model=KMeans(n_clusters=3, random_state=0).fit(data1)

clusterbymodel= model.predict(data1)
data1["label"] = clusterbymodel

#data1.loc["group"] = clusterbymodel
np.append(data1,clusterbymodel)
data1
clusterbymodel.shape
data1=data1[:1000]
from scipy.cluster.hierarchy import dendrogram, linkage

def dendrogram_result(Z, title, xlabel, ylabel):

    fig = plt.figure(figsize=(20, 100))

    dendrogram(Z, labels=None, leaf_rotation=0, orientation="left")

    plt.title(title,fontsize=30)

    plt.xlabel(xlabel,fontsize=30)

    plt.ylabel(ylabel,fontsize=30)
Z = linkage(data1, 'complete')

title = 'Clustering in airbnb NYC by average price of each neighbourhood'

ylabel = 'Neighbourhood'

xlabel = 'Price'



dendrogram_result(Z, title, xlabel, ylabel)
data1