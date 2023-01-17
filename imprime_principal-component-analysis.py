import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print(cancer['DESCR'])
print(cancer['target_names'])
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

#(['DESCR', 'data', 'feature_names', 'target_names', 'target'])
df.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(df)
scaled_data = scaler.transform(df)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape
plt.figure(figsize=(8,6))

plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')

plt.xlabel('First principal component')

plt.ylabel('Second Principal Component')
pca.components_
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
plt.figure(figsize=(12,6))

sns.heatmap(df_comp,cmap='plasma',)