import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
dataset = pd.read_csv("../input/fashion-mnist_test.csv")
X = dataset.iloc[:, 1:]
y = dataset.iloc[:, :1]
print(X)
print(y)
pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)

plt.plot(range(4), pca.explained_variance_ratio_)
plt.plot(range(4), np.cumsum(pca.explained_variance_ratio_))
plt.title("Component-wise and Cumulative Explained Variance")
X_r.shape