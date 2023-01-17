# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/glass.csv')

df
columns_names=df.columns.tolist()

print("Columns names:")

print(columns_names)
df.shape

df.head()
df.corr()
correlation = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')

X = df.iloc[:,0:11].values

y = df.iloc[:,0].values

X



np.shape(X)
y

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



    plt.bar(range(10), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()



significance = [np.abs(i)/np.sum(eig_vals) for i in eig_vals]



#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(significance))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Explained Variance')

plt.show()
matrix_w = np.hstack((eig_pairs[0][1].reshape(10,1), 

                      eig_pairs[1][1].reshape(10,1),

                      eig_pairs[2][1].reshape(10,1),

                      eig_pairs[3][1].reshape(10,1),

                      eig_pairs[4][1].reshape(10,1)

                    ))

print('Matrix W:\n', matrix_w)



np.shape(matrix_w)
Y = X_std.dot(matrix_w)

Y