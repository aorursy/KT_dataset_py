# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/winemag-data-130k-v2.csv")
df = df[['country', 'points', 'price']].fillna(value=0)
for index, row in df.iterrows():
    if row['price'] == 0.0:
        df.drop(index, inplace=True)
df    
def partition_dataset(data):
    
    x = data[['points', 'price']][:-1300]
    y = data[['country']][:-1300]
    x_test = data[['points', 'price']][-1300:]
    y_test = data[['country']][-1300:]
    
    assert type(data) == pd.core.frame.DataFrame
    return x, y, len(y_test.groupby(by='country')), x_test, y_test
x, y_true, num_clusters, x_test, y_test = partition_dataset(df)
print("Num clusters: ", num_clusters)
k_means = KMeans(n_clusters=num_clusters)
k_means.fit(x)
y_predict = k_means.predict(x)
print(k_means.score(x_test, y_test)) 
labels = k_means.labels_
cluster_centers = k_means.cluster_centers_
plt.figure(figsize=(30,30))
plt.scatter(x['points'], x['price'], c=y_predict, s=40, cmap='gist_rainbow', alpha=.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=100, cmap='viridis', alpha=.7);