## Importing necessary modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/nndb_flat.csv")
df.head()
df.columns
df.shape
correlation = df.corr()

correlation
df.head()
### Drop 

df_drop = df.drop(labels=['ID','FoodGroup','ShortDescrip','Descrip','CommonName','MfgName','ScientificName'],axis=1)
df_drop.head()
df_drop.shape
X = df_drop.iloc[:,1:37].values

y = df_drop.iloc[:,0].values

X
y
np.shape(X)
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
## Covariance Matrix

mean_vec = np.mean(X_std, axis=0)

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
eig_vals, eig_vecs = np.linalg.eig(cov_mat)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
# Make a list of (eigenvalue, eigenvector) tuples

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

eig_pairs.sort(key=lambda x: x[0], reverse=True)



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:')

for i in eig_pairs:

    print(i[0])
tot = sum(eig_vals)

var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))



    plt.bar(range(36), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()

matrix_w = np.hstack((eig_pairs[0][1].reshape(36,1), 

                      eig_pairs[1][1].reshape(36,1)

                    ))

print('Matrix W:\n', matrix_w)
Y = X_std.dot(matrix_w)

Y
## PCA IN SCIKIT LEARN

from sklearn.decomposition import PCA

pca = PCA().fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,36,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
from sklearn.decomposition import PCA 

sklearn_pca = PCA(n_components=20)

Y_sklearn = sklearn_pca.fit_transform(X_std)
print(Y_sklearn)
Y_sklearn.shape