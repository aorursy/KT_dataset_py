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
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df = df.drop(columns = ['id','name','host_id','host_name','latitude','longitude','last_review','neighbourhood'])
df = pd.get_dummies(df)
df.isna().mean()
df[df.isna()['reviews_per_month']]['number_of_reviews'].value_counts()
df['reviews_per_month'] = df['reviews_per_month'].fillna(df['number_of_reviews'])
from sklearn.preprocessing import normalize

data_scaled = normalize(df)

data_scaled = pd.DataFrame(data_scaled, columns=df.columns)

data_scaled.head()
data_scaled = data_scaled[0:1000]
import scipy.cluster.hierarchy as shc

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")  

plt.axhline(y=2.5, color='r', linestyle='--')

dend = shc.dendrogram(shc.linkage(data_scaled[0:100], method='ward'))
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  

final_cluster = cluster.fit_predict(data_scaled)
data_scaled['cluster'] = final_cluster
data_scaled