import pandas as pd
import numpy as np
df = pd.read_csv('../input/Mall_Customers.csv')
df.head()
df.shape
df.isnull().sum()
import seaborn as sns
%config InlineBackend.figure_format = 'png'
sns.pairplot(df[['CustomerID', 'Age', 'Annual Income (k$)']]);
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

kmeans = KMeans(n_clusters=5, random_state=1)
numeric_cols = df._get_numeric_data().dropna(axis=1)
kmeans.fit(numeric_cols)


# Visualizing using PCA
pca = PCA(n_components=2)
res = pca.fit_transform(numeric_cols)
plt.figure(figsize=(12,8))
plt.scatter(res[:,0], res[:,1], c=kmeans.labels_, s=50, cmap='viridis')
plt.title('PCA');
