# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install factor-analyzer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA

from factor_analyzer import FactorAnalyzer
%matplotlib inline
# Load the train data
train=pd.read_csv('../input/instant-gratification/train.csv')
train.head()
# drop the id and target col
df=train.drop(columns=['target','id'])
df.dtypes
# StandardScaler + PCA
scaler = StandardScaler()
train_scaled = scaler.fit_transform(df)     

pca = PCA(n_components=4)
PCA_train_x = pca.fit_transform(train_scaled)

plt.scatter(PCA_train_x[:, 0],PCA_train_x[:, 1],
            c=train.target,cmap="copper_r")
plt.colorbar()
plt.show()
# Loadings
loadings=pd.DataFrame(pca.components_.T,columns=['PC1','PC2', 'PC3', 'PC4'],
                      index=df.columns)
print(loadings.sort_values(by=['PC1']))
print(loadings.sort_values(by=['PC2']))
print(loadings.sort_values(by=['PC3']))
print(loadings.sort_values(by=['PC4']))


# PCA loadings are the coefficients of the linear combination of 
# the original variables from which the principal components (PCs)
# are constructed.
# Loadings Matrix
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4'],
                              index=df.columns)

print(loading_matrix.sort_values(by=['PC1']))
print(loading_matrix.sort_values(by=['PC2']))
print(loading_matrix.sort_values(by=['PC3']))
print(loading_matrix.sort_values(by=['PC4']))
fa = FactorAnalyzer(18, rotation='varimax',
                    method='principal',impute='mean')
fa.fit(train_scaled)

ev, v = fa.get_eigenvalues()
print(ev)

#Create scree plot using matplotlib
plt.figure(figsize=(8, 6.5))
plt.scatter(range(1,train_scaled.shape[1]+1),ev)
plt.plot(range(1,train_scaled.shape[1]+1),ev)
plt.title('Scree Plot',fontdict={'weight':'normal','size': 25})
plt.xlabel('Factors',fontdict={'weight':'normal','size': 15})
plt.ylabel('Features',fontdict={'weight':'normal','size': 15})
plt.grid()
plt.show()

# no. of factors
n_factors = sum(ev>1)
fa2 = FactorAnalyzer(n_factors,rotation='varimax',method='principal')
fa2.fit(train_scaled)

var = fa2.get_factor_variance()

fa2_score = fa2.transform(train_scaled)

column_list = ['fac'+str(i) for i in np.arange(n_factors)+1]
fa_score = pd.DataFrame(fa2_score,columns=column_list)
for col in fa_score.columns:
    data[col] = fa_score[col]
print("\n Factor Scores:\n",fa_score)    

df_fv = pd.DataFrame()
df_fv['Factors'] = column_list
df_fv['Contribution'] = var[1]
df_fv['Acc. Contribution'] = var[2]
df_fv['Acc. Contribution Pct'] = var[1]/var[1].sum()
print("\n list:\n",df_fv) 
# Kernel PCA
lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), 
                            (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
       
    PCA_train_x = PCA(2).fit_transform(train_scaled)
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(PCA_train_x[:, 0], PCA_train_x[:, 1], c=train.target, cmap="viridis")
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

plt.show()
# K Means
kmeans=KMeans(n_clusters=2,random_state=0).fit(PCA_train_x)
y_kmeans = kmeans.predict(PCA_train_x)
centers=kmeans.cluster_centers_
# Remake the Plot
plt.scatter(PCA_train_x[:, 0],PCA_train_x[:, 1],
            c=y_kmeans,cmap="copper_r")
# plt.axis('off')
plt.colorbar()
plt.show()