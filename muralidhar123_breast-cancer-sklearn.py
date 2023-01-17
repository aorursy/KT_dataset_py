import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import seaborn as sns
%matplotlib inline
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print(cancer['feature_names'])
print(cancer['DESCR'])
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler_data = scaler.fit_transform(df)
scaler_data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca
x_pca = pca.fit_transform(scaler_data)
x_pca
scaler_data.shape
x_pca.shape
plt.figure(figsize=(8,6))

plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')

plt.xlabel('first principle component')

plt.ylabel('second principle component')

plt.show()