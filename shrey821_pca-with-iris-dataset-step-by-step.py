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
# import libs

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

import os

import missingno as msno
# loading iris dataset

iris_data = pd.read_csv("/kaggle/input/iris/Iris.csv",index_col='Id')

iris_data.info()
# getting some insights of the data

iris_data.describe()
iris_data.head()
# converting it to the species col to the matching values for the species

iris_data.Species.replace({'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2},inplace=True)

iris_data.head()
# visualization for the data

p=msno.bar(iris_data)
sns.countplot(y=iris_data.Species ,data=iris_data)

plt.xlabel("Count of each Target class")

plt.ylabel("Target classes")

plt.show()
# individual ploting for features 

fig,ax = plt.subplots(nrows = 2, ncols=2, figsize=(16,10))

row = 0

col = 0

for i in range(len(iris_data.columns) -1):

    if col > 1:

        row += 1

        col = 0

    axes = ax[row,col]

    sns.boxplot(x = iris_data['Species'], y = iris_data[iris_data.columns[i]],ax = axes)

    col += 1

plt.tight_layout()

plt.title("Individual Features by Class")

plt.show()
p=sns.pairplot(iris_data, hue = 'Species')
# heatmap plot for the correlation

plt.figure(figsize=(15,15))

p=sns.heatmap(iris_data.corr(), annot=True,cmap='RdYlGn')
# plots for the features distribution

iris_data.hist(figsize=(15,12),bins = 15)

plt.title("Features Distribution")

plt.show()
X = iris_data.drop(['Species'],axis=1)

y = iris_data.Species

pca = PCA()

X_new = pca.fit_transform(X)
# co-variance matrix

pca.get_covariance()
# getting variance ratio

explained_variance=pca.explained_variance_ratio_

print(explained_variance)
# plot with individual explained variance and principal components

with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))



    plt.bar(range(4), explained_variance, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
pca=PCA(n_components=3)

X_new=pca.fit_transform(X)

X_train_new, X_test_new, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state=20, stratify=y)
# performing knn based pca over it

knn_pca = KNeighborsClassifier(7)

knn_pca.fit(X_train_new,y_train)

print("Train score after PCA",knn_pca.score(X_train_new,y_train),"%")

print("Test score after PCA",knn_pca.score(X_test_new,y_test),"%")
# Visualising the Test set results

classifier = knn_pca

from matplotlib.colors import ListedColormap

X_set, y_set = X_test_new, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel(),np.zeros((X1.shape[0],X1.shape[1])).ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('KNN PCA (Test set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()