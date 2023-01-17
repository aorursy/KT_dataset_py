from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

data.keys()
data["target"]
import pandas as pd
df = pd.DataFrame(data["data"], columns=data["feature_names"])

df.head()
df.info()
df.describe()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
import numpy as np
#Singular value decomposition

U, gamma, V = np.linalg.svd(scaled_data, full_matrices=False)
gamma
gamma.shape
Gamma = np.diag(gamma)

Gamma.shape
import matplotlib.pyplot as plt
plt.figure(figsize=(18, 10))

plt.scatter(range(1,len(gamma)+1), gamma)

plt.grid()

plt.ylabel("Singular Values")

plt.xlabel("feature index i")

plt.title("Singular values plot")
# principal component

from sklearn.decomposition import PCA

# n_components = number of components after PCA

pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
# Only 2 dimensions!

x_pca.shape
xpca_df = pd.DataFrame(x_pca, columns=["PC1", "PC2"])

xpca_df.head()
xpca_df.info
# Plot the 2d space, orange dots are benign cells



import seaborn as sns



plt.figure(figsize=(12,8))

sns.scatterplot(x_pca[:,0], x_pca[:,1], hue=data["target"], s=60)



plt.ylabel("First Principal Component")

plt.xlabel("Second Principal Component")

plt.title("Input Data in the Reduced Feature Space")

plt.xlim([-10,20])

plt.xlim([-10,15])
# 1st array: 1st component weights

# 2nd array: 2nd component weights

pca.components_
df_comp = pd.DataFrame(pca.components_, columns=data["feature_names"])

df_comp.index = ["1st component", "2nd component"]

df_comp
plt.figure(figsize=(12,8))



sns.heatmap(df_comp, cmap="autumn")