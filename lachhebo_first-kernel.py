import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA
df_distro = pd.read_csv("../input/Distrowatch_popularity.csv", index_col = "Unnamed: 0")
df_distro.head()
f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)

sns.despine(left=True)



sns.distplot(df_distro["12_months"],  ax=axes[(0, 0)])

sns.distplot(df_distro["6_months"],  ax=axes[(0, 1)])

sns.distplot(df_distro["3_months"],  ax=axes[(1, 0)])

sns.distplot(df_distro["1_month"],  ax=axes[(1, 1)])



plt.setp(axes, yticks=[])

plt.tight_layout()

pca = PCA(n_components = 2)



X_pca = pca.fit_transform(df_distro[["12_months","6_months","3_months","1_month"]])



plt.subplots(figsize=(15, 15))



plt.scatter(X_pca[:,0],X_pca[:,1]);

plt.xlabel("first principal component")

plt.xlabel("second principal component")

plt.title("PCA of linux distribution")
