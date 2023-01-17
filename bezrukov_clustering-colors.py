# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import data from csv
df = pd.read_csv("../input/colors.csv")
df.info()
df.head()
# create a function that performs dimentionality reduction and visualization
def visualize_in_2D(df, marker, size):
    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(df.loc[:, "R":"B"]))

    color = []
    for i in range(df.shape[0]):
        color.append(tuple(df.loc[i, "R":"B"]))
    plt.scatter(df_pca[0], df_pca[1], color=color, marker = marker, s=size)

visualize_in_2D(df, "o", 20)
# perform k-means clustering for a range of n_clusters and calculate the corresponding inertia
inertia_list = []

for i in range(2, 15):
    model = KMeans(n_clusters = i)
    model.fit(df.loc[:, "R":"B"])
    inertia_list.append(model.inertia_)

# plot inertia as the function of n_clusters
plt.plot(range(2, 15), inertia_list, marker="o")
# create a model for k-means clustering with optimal n_clusters
n_clusters = 4
model = KMeans(n_clusters = n_clusters)

# cluster colors to groups and pass group labels to a new column in df
df["group_label"] = model.fit_predict(df.loc[:, "R":"B"])

# find group centers
group_centers = pd.DataFrame(model.cluster_centers_, columns=["R", "G", "B"])

print("Centers for each group:")
print(group_centers)
# plot representative colors for each group
for i in range(n_clusters):
    color = tuple(group_centers.iloc[i, :])
    plt.scatter(i*3, 1, color=color, marker = "s", s=3000)
visualize_in_2D(df, "o", 20)
visualize_in_2D(group_centers, "+", 800)