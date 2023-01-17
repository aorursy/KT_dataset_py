import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
print(scale.fit(df))
scale_data = scale.transform(df)
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
print(pca.fit(scale_data))
s_pca = pca.transform(scale_data)
print(scale_data.shape)
print(s_pca.shape)
plot.figure(figsize = (8,8))
plot.scatter(s_pca[:,0],s_pca[:,1],c=cancer['target'],cmap='rainbow')
plot.xlabel('First principal component')
plot.ylabel('Second principal component')