import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
dataset = pd.read_csv("../input/HR_comma_sep.csv")
dataset
column_names = dataset.columns.tolist()
print(column_names)
correlation = dataset.corr()
correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
dataset['Department'].unique()
X = dataset.iloc[:,1:8].values
y = dataset.iloc[:,0].values
X
y
np.shape(X)
np.shape(y)
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')