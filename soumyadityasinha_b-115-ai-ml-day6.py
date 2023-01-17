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
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



iris_data = pd.read_csv("../input/iris/Iris.csv",index_col='Id')

iris_data.Species.replace({'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2},inplace=True)



X = iris_data.drop(['Species'],axis=1)

y = iris_data.Species









from sklearn.decomposition import PCA

pca = PCA()

X_new = pca.fit_transform(X)



pca.get_covariance()



explained_variance=pca.explained_variance_ratio_

explained_variance



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