import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/hr-comma-sepcsv/HR_comma_sep.csv')




df.corr()



correlation = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')
df_drop=df.drop(labels=['sales','salary'],axis=1)

df_drop.head()
cols = df_drop.columns.tolist()

cols
cols.insert(0, cols.pop(cols.index('left')))





df_drop = df_drop.reindex(columns= cols)



X = df_drop.iloc[:,1:8].values

y = df_drop.iloc[:,0].values

X
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))




plt.figure(figsize=(8,8))

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



    plt.bar(range(7), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1), 

                      eig_pairs[1][1].reshape(7,1)

                    ))

print('Matrix W:\n', matrix_w)
Y = X_std.dot(matrix_w)

Y
########################3



from sklearn.decomposition import PCA

pca = PCA().fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,7,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
from sklearn.decomposition import PCA 

sklearn_pca = PCA(n_components=6)

Y_sklearn = sklearn_pca.fit_transform(X_std)





print(Y_sklearn)


