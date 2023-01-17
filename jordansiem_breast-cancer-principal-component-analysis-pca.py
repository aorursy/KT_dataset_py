#PCA attempts to figure out which features explain the most variance
import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

%matplotlib inline
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print(cancer['DESCR'])
#Not predicting trying to figure out what is cause of most variance
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()
cancer['target_names']
from sklearn.preprocessing import StandardScaler
#pca to have 2 dimensions instead of 30.
#Scale data to have single unit variance
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
#PCA - decomposing into principal componenets

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

#number of components you want to keep
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape
#reduced 30 to 2
plt.figure(figsize=(8,6))

plt.scatter(x_pca[:,0],x_pca[:,1])

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component')

#plot all rows in column 0 vs. rows in column 1
plt.figure(figsize=(8,6))

plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component')

#color by malignant and benign
#Off two components we see seperation of each. vs. 30 compression algo
#What is 1st pca and 2nd pca? Not one to one....correspond to combinations
#of the original features. stored as attribute of original object
pca.components_
#row - pca col - original features. use heatmap to visualize
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
df_comp
#data frame pca 0 and pca 1 and relationship of each features
plt.figure(figsize=(12,6))

sns.heatmap(df_comp,cmap='plasma')
#relationship of correlation of each feature and pca themselves
#each pca shown as row. hotter colors is more correlated to specific features
#combo of all features
#Use for high dimensional data.