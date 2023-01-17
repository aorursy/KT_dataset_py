import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

cancer_data.keys()
CC = pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])

CC.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(CC)
scaled_data = scaler.transform(CC)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape
plt.figure(figsize=(8,6))

plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])

plt.xlabel('First principal component')

plt.ylabel('Second Principal Component')