# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/StudentsPerformance.csv")
data.head()
x = data.loc[:, "math score":].values

from sklearn.decomposition import PCA
pca = PCA(n_components = 2, whiten= True) #witten = normalize
pca.fit(x)

x_pca = pca.transform(x)
print("Accuracy of PCA : ", sum(pca.explained_variance_ratio_))
df = pd.DataFrame(dict(p1 = x_pca[:, 0], p2 = x_pca[:, 1] ))
data = pd.concat([data,df], axis=1)
import seaborn as sns
g = sns.pairplot(data.loc[:,["gender", "p1", "p2"]], hue="gender",plot_kws=dict(alpha=0.4))
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
clusters = kmeans.fit_predict(df)
import matplotlib.pyplot as plt

df["clusters"] = clusters
data["clusters"] = clusters

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(17, 5))

ax1.scatter(data.p1[data.gender == "female"], data.p2[data.gender == "female"], alpha=0.5)
ax1.scatter(data.p1[data.gender == "male"], data.p2[data.gender == "male"], alpha=0.5)
ax1.set_title('Accurate Data')

ax2.scatter(df.p1, df.p2, alpha=0.6, color="black")
ax2.set_title('Data Without Clustering')

ax3.scatter(df.p1[df.clusters == 0], df.p2[df.clusters == 0], alpha=0.5)
ax3.scatter(df.p1[df.clusters == 1], df.p2[df.clusters == 1], alpha=0.5)
ax3.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color="yellow", s=150)
ax3.set_title('Data With K-Means Clustering Algorithm')

plt.show()