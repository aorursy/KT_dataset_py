import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv("../input/wine-customer-segmentation/Wine.csv")

dataset
dataset.info()
import numpy as np

import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer.keys()
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

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
#Using scatter plot show where the Principal components lie on the graph. 

plt.figure(figsize=(8,6))

plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='rainbow')

plt.xlabel('First principal component')

plt.ylabel('Second Principal Component')