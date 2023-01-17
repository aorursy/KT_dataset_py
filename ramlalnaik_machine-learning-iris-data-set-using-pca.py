# Suprass warnings

import warnings

warnings.filterwarnings
# Import required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# Import iris dataset

iris = pd.read_csv("../input/iris-datasets/Iris.csv")

iris.head()
# Checking basic information

iris.info()
#Checking descriptive statistics

iris.describe().T
# Checking the shape of the dataset

iris.shape
# Do not take non numerical or boolean data as inputs

iris.Species.replace({'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2},inplace=True)
#After conversion of non numerical data

iris.head()
# Removing unwanted columns -"ID"

iris.drop('Id', axis = 1, inplace =True)
# Checking null count

import missingno as msno

p=msno.bar(iris)
# Ploting pairplot for dataset

sns.pairplot(iris, hue ='Species')
# Boxplot

fig,ax = plt.subplots(nrows = 2, ncols=2, figsize=(16,10))

row = 0

col = 0

for i in range(len(iris.columns) -1):

    if col > 1:

        row += 1

        col = 0

    axes = ax[row,col]

    sns.boxplot(x = iris['Species'], y = iris[iris.columns[i]],ax = axes)

    col += 1

plt.tight_layout()

plt.show()
# Checking Correlation

plt.figure(figsize =(12,6))

sns.heatmap(iris.corr(), annot= True)
# creating X and y

X = iris.drop(['Species'], axis =1)

y = iris.Species
# Importing Scaler

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
# Splitting dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size =0.7, test_size =0.3, random_state =42)
#Checking shape for test and train dataset

X_train.shape,  X_test.shape
# Importing PCA package

from sklearn.decomposition import PCA

pca = PCA()

X_new = pca.fit_transform(X)
# Creating coavariance matrix

cov_mat = np.cov(X_new.T)

print('Covariance Matrix', cov_mat)
#Creating eigen vectors & eigen Values



eig_val, eig_vect = np.linalg.eig(cov_mat)

print('Eigen Values',eig_val)

print('Eigen Vectors', eig_vect)
#sort eigenvalues in decending order

eig_pairs = [(np.abs(eig_val[i]), eig_vect[:, i]) for i in range (len(eig_val))]

tot = sum(eig_val)

var_exp = [(i/tot)*100 for i in sorted (eig_val, reverse =True)]

print('Cummulative Variance Explained', var_exp)
# Barplot



plt.figure(figsize=(6, 4))



plt.bar(range(4), var_exp, alpha=0.5, align='center', label='individual explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.tight_layout()

plt.show()
#Importing required packages

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from sklearn import datasets
# reloading dataset

iris = datasets.load_iris()

X = iris.data

y = iris.target
# to getter a better understanding of interaction of the dimensions

#plot the first three PCA dimensions

fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=110)

X_reduced =PCA(n_components=3).fit_transform(iris.data)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:,2], c=y, cmap =plt.cm.Paired)



ax.set_title ("First three PCA directions")

ax.set_xlabel("First Eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("Second Eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("Third Eigenvector")

ax.w_zaxis.set_ticklabels([])



plt.show()