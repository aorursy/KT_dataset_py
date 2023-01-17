# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
### Pegando os dados do kaggle

dadosOriginais = pd.read_csv("/kaggle/input/glass/glass.csv")



## Cópia

dadosGlass = dadosOriginais
### Correlação de Peason

corr = dadosGlass.corr(method='pearson')

corr
### Correlação pintada

#corr = dadosGlass.corr(method='pearson')

import matplotlib.pyplot as plt

import seaborn as sns

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);

### Mg e Type (-0.74): 

### Ca com RI (0.81)
### Correlação pintada e com valores

#corr = dadosGlass.corr(method='pearson')

#plt.figure(figsize=(10,10))

#sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='cubehelix')
# PCA

# transferir colunas como o label e nao como uma feature

## linhas como coluna

cols = dadosGlass.columns.tolist()

cols.insert(0, cols.pop(cols.index('Type')))

dadosGlass = dadosGlass.reindex(columns= cols)

##

X = dadosGlass.iloc[:,1:10].values

y = dadosGlass.iloc[:,0].values
# Data Standardisation

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)

## Covariance matrix



mean_vec = np.mean(X_std, axis=0)

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)



#print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
### Covariancia colorida

plt.figure(figsize=(10,10))

sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')
##Eigen decomposition of the covariance matrix

#https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix



eig_vals, eig_vecs = np.linalg.eig(cov_mat)

#print('Eigenvectors \n%s' %eig_vecs)

#print('\nEigenvalues \n%s' %eig_vals)
# Selecting Principal Components

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



    plt.bar(range(9), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
matrix_w = np.hstack((eig_pairs[0][1].reshape(9,1), 

                      eig_pairs[1][1].reshape(9,1)

                    ))

print('Matrix W:\n', matrix_w)
Y = X_std.dot(matrix_w)

Y
### PCA in scikit-learn: https://www.kaggle.com/nirajvermafcb/principal-component-analysis-with-scikit-learn

from sklearn.decomposition import PCA

pca = PCA().fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,9,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')

from sklearn.decomposition import PCA 

sklearn_pca = PCA(n_components=6)

Y_sklearn = sklearn_pca.fit_transform(X_std)

#print(Y_sklearn)

Y_sklearn.shape