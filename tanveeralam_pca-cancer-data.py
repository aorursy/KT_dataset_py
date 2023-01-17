import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
from sklearn.datasets import load_breast_cancer
raw_data = load_breast_cancer()
type(raw_data)
raw_data.keys()
print(raw_data['DESCR'])
print(raw_data['target'])
df = pd.DataFrame(raw_data['data'], columns=raw_data['feature_names'])
df.head(3).T

df.shape
df.isnull().sum()
df.describe().T
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_df = scaler.transform(df)
scaled_df[0]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_df)

pca_df = pca.transform(scaled_df)
scaled_df.shape, pca_df.shape
plt.figure(figsize=(12,6))
plt.scatter(pca_df[:, 0], pca_df[:, 1], c=raw_data['target'], cmap='plasma_r')
plt.xlabel('First Feature')
plt.ylabel('Second Feature')
plt.title('1st Feature vs 2nd Feature', fontsize=20, weight='bold')
pca.components_
df_componets = pd.DataFrame(pca.components_, columns=raw_data['feature_names'])
df_componets.head(4)
plt.figure(figsize=(12,6))
sns.heatmap(df_componets)
