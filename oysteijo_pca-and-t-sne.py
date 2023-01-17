# Import the important libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# Read the data

df = pd.read_csv("../input/train.csv").set_index("ID")
desc = df.describe().transpose()

columns_to_drop = desc.loc[desc["std"]==0].index.values

df.drop(columns_to_drop, axis=1, inplace=True)
print(columns_to_drop)
df08 = df[["X{}".format(x) for x in range(9) if x != 7]]
tot_cardinality = 0

for c in df08.columns.values:

    cardinality = len(df08[c].unique())

    print(c, cardinality)

    tot_cardinality += cardinality

print(tot_cardinality)
df = pd.get_dummies(df, columns=["X{}".format(x) for x in range(9) if x != 7])
# sns.distplot(df.y)

#(Why do I get a warning?)

# I get a long warning on the kaggle kernel, I'm commenting this line.
# Drop it!

df.drop(df.loc[df["y"] > 250].index, inplace=True)
from sklearn.decomposition import PCA

pca2 = PCA(n_components=2)

pca2_results = pca2.fit_transform(df.drop(["y"], axis=1))
cmap = sns.cubehelix_palette(as_cmap=True)

f, ax = plt.subplots(figsize=(20,15))

points = ax.scatter(pca2_results[:,0], pca2_results[:,1], c=df.y, s=50, cmap=cmap)

f.colorbar(points)

plt.show()
from sklearn.manifold import TSNE

tsne2 = TSNE(n_components=2)

tsne2_results = tsne2.fit_transform(df.drop(["y"], axis=1))
f, ax = plt.subplots(figsize=(20,15))

points = ax.scatter(tsne2_results[:,0], tsne2_results[:,1], c=df.y, s=50, cmap=cmap)

f.colorbar(points)

plt.show()