# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#Loading the dataset
df = pd.read_csv('../input/HR_comma_sep.csv')
# Any results you write to the current directory are saved as output.
columns_names=df.columns.tolist()
print("Columns names:")
print(columns_names) #list all columns

df.shape
df.head()
#df.corr() compute pairwise correlation of columns.
#Correlation shows how the two variables are related to each other.Positive values shows as one variable increases other variable increases as well. 
#Negative values shows as one variable increases other variable decreases.Bigger the values,more strongly two varibles are correlated and viceversa.
correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different fearures')
#Principal Component Analysis
df.head()
df_drop=df.drop(labels=['salary','Department'],axis=1)
df_drop.head()
cols = df_drop.columns.tolist()
cols
df_drop = df_drop.reindex(columns= cols)
X = df_drop.iloc[:,1:8].values
y = df_drop.iloc[:,0].values
X
np.shape(X)

np.shape(y)
#Data Standardisation - Standardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance). 
#It is useful to standardize attributes for a model. 
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
#PCA in scikit-learn
from sklearn.decomposition import PCA
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


#The above plot shows almost 90% variance by the first 6 components. Therfore we can drop 7th component.
from sklearn.decomposition import PCA 
sklearn_pca = PCA(n_components=6)
Y_pca = sklearn_pca.fit_transform(X_std)
Y_pca.shape

