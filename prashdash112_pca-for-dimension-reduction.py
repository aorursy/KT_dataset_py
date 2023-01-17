import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
type(cancer)
cancer.keys()
#cancer.values()
print(cancer['DESCR'])
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head(5)
cancer['target']
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scale.fit(df)
for i in cancer['feature_names']:

    print(i);
np.std(df['worst area'])
df['worst area'].plot(kind='kde',figsize=(12,6))
scaled_data=scale.transform(df)
scaled_data
#PCA

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
scaled_data.shape
x_pca.shape

#dimension_reductionality from 30 to 2
plt.figure(figsize=(10,6))

plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])

plt.xlabel('first principal component')

plt.ylabel('second principal component')

plt.legend()
print(pca.components_)
df_comp=pd.DataFrame(data=pca.components_ , columns=cancer['feature_names'])
df_comp
plt.tight_layout()

df_comp.plot(figsize=(12,6))
plt.figure(figsize=(12,7))

sns.heatmap(df_comp)