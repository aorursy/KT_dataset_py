# Let us import the required packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns # visualize the co-relation using this library

%matplotlib inline

from sklearn.preprocessing import StandardScaler
# Let us read the data from the file and see the first five rows of the data

data = pd.read_csv("../input/DS_BEZDEKIRIS_STD.data", header = None)

data.head()
data.corr()
correlation = data.corr()

plt.figure(figsize=(4,4))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')
X = data.iloc[:,0:4].values

y = data.iloc[:,-1].values

X
y
np.shape(X)
np.shape(y)
X_std = StandardScaler().fit_transform(X)
X_std.shape
mean_vec = np.mean(X_std, axis=0)

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
plt.figure(figsize=(4,4))

sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different features')
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



    plt.bar(range(4), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
feature_vector = np.hstack((eig_pairs[0][1].reshape(4,1), 

                      eig_pairs[1][1].reshape(4,1)

                    ))

print('Matrix W:\n', feature_vector)
final_data = X_std.dot(feature_vector)
final_data
from sklearn.decomposition import PCA

pca = PCA().fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,4,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
from sklearn.decomposition import PCA 

sklearn_pca = PCA(n_components=2)

Y_sklearn = sklearn_pca.fit_transform(X_std)
Y_sklearn