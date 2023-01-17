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
from pandas import read_csv

data = read_csv('../input/glass.csv')

X = data.drop("Type",axis= 1)

y = data["Type"]

X.head()
X.corr()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

correlation = X.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, annot=True)



plt.title('Correlation between different fearures')
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T)) #using its package, numpy helps us directly find the covar 

cov_matrix= np.cov(X_std.T)

print(cov_matrix.shape)

print(X.shape)

plt.figure(figsize=(8,8))

sns.heatmap(cov_matrix, annot=True)



plt.title('Correlation between different features')
eigen_vals, eigen_vects = np.linalg.eig(cov_matrix)



print('Eigenvectors \n%s' %eigen_vects)

print('\nEigenvalues \n%s' %eigen_vals)
# Make a list of (eigenvalue, eigenvector) tuples

eigen_pairs = [(eigen_vals[i], eigen_vects[:,i]) for i in range(len(eigen_vals))]

eigen_pairs



# Sort the (eigenvalue, eigenvector) tuples from high to low

eigen_pairs.sort(key=lambda x: x[0], reverse=True)
total= sum(eigen_vals)

#this will helps us while finding the amount of data preserved in %



preserved_percent= [(i/ total)* 100  for i in sorted(eigen_vals, reverse= True)]

preserved_percent
plt.figure(figsize=(6, 4))

plt.bar(range(9), preserved_percent, alpha=0.5, align='center', label='individual explained variance')

plt.ylabel('variance percentages')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.tight_layout()
matrix_v= np.hstack((eigen_pairs[0][1].reshape(9, 1), eigen_pairs[1][1].reshape(9,1), eigen_pairs[2][1].reshape(9, 1)))

#hstack stacks arrays horizontally

matrix_v
Y= X_std.dot(matrix_v)

#dot product of the 2 vectors

print(Y)