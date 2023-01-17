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

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import cv2

import warnings

from imblearn.over_sampling import KMeansSMOTE

warnings.filterwarnings('ignore')
df = np.load('/kaggle/input/eval-lab-4-f464/train.npy',allow_pickle=True)

test = np.load('/kaggle/input/eval-lab-4-f464/test.npy',allow_pickle=True)
names = []

images = []

for row in df:

    names.append(row[0])

    images.append(row[1])
plt.imshow(images[2], cmap='gray')

plt.show()
# def distance(x1, x2):

#     return np.sqrt(((x1-x2)**2).sum())



# # For each k in k_values return majority label

# def knn_batch_k(X_train, y_train, test_sample, k_values):

#     distances = []

#     for i in range(len(X_train)):

#         distances.append([distance(X_train[i], test_sample), y_train[i]])

#     distances.sort()

#     return [pd.DataFrame(distances[:k])[1].value_counts().idxmax() for k in k_values]



# # Return class of each test sample predicted by knn for each k in k_value.

# def predict_batch_k(X_train, y_train, X_test, k_values=np.arange(2,10)):

#     y_pred = []

#     for test_sample in X_test:

#         y_pred.append(knn_batch_k(X_train, y_train, test_sample, k_values))

#     return y_pred
flat_images = []

for img in images:

    flat_images.append(img.reshape((1,50*50*3)))
test_flat_images = []

for row in test:

    test_flat_images.append(row[1].reshape((1,50*50*3)))
# gray_image = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
# images_gray=[]

# for img in images:

#     images_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
# images_gray = np.array(images_gray)

# images_gray = images_gray.reshape((2275,2500))
# from sklearn.neighbors import KNeighborsClassifier

# knn_clf = KNeighborsClassifier(n_neighbors=3)

# knn_clf.fit(images_gray, names)
pd.DataFrame(names)[0].unique()
test_images_gray = []

test_flat_images = []

for row in test:

    test_flat_images.append(row[1])

    test_images_gray.append(cv2.cvtColor(row[1], cv2.COLOR_BGR2GRAY))
# test_images_gray = np.array(test_images_gray)

# test_images_gray = test_images_gray.reshape((976,2500))
plt.imshow(test_flat_images[12], cmap='gray')

plt.show()
flat_images = np.array(flat_images)

flat_images = flat_images.reshape((2275,7500))
test_flat_images = np.array(test_flat_images)

test_flat_images = test_flat_images.reshape((976,7500))
# y_pred = knn_clf.predict(test_images_gray)
# from sklearn.cluster import DBSCAN

# clt = DBSCAN(metric="euclidean")

# clt.fit(images_gray)
# from kmodes.kmodes import KModes

# from kmodes.kprototypes import KPrototypes



# km = KModes(n_clusters=19, init='Huang',max_iter=1000000 ,n_init=5, verbose=1,random_state=0)

# clusters1 = km.fit_predict(test_images_gray)
import os

from gzip import GzipFile



import numpy as np

import pylab as pl

from sklearn.decomposition import PCA as RandomizedPCA



n_components = 85

X_train = flat_images

pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

X_test = test_flat_images



X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn.svm import SVC



param_grid = {

 'C': [1, 5, 10, 50, 100],

 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1,'scale'],

}

clf = GridSearchCV(SVC(kernel='rbf'), param_grid)

clf = clf.fit(X_train_pca, names)
y_pred = clf.predict(X_test_pca)

print(y_pred)
out = pd.DataFrame(data={'ImageId':np.arange(0,976),'Celebrity':y_pred})

out.to_csv('submission_model1.csv',index=False)
import os

from gzip import GzipFile



import numpy as np

import pylab as pl

from sklearn.decomposition import PCA as RandomizedPCA



n_components = 80

X_train = flat_images

pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

X_test = test_flat_images



X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn.svm import SVC



param_grid = {

 'C': [1, 5, 10, 50, 100],

 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1,'scale'],

}

clf = GridSearchCV(SVC(kernel='rbf'), param_grid)

clf = clf.fit(X_train_pca, names)
y_pred = clf.predict(X_test_pca)

print(y_pred)
out = pd.DataFrame(data={'ImageId':np.arange(0,976),'Celebrity':y_pred})

out.to_csv('submission_model2.csv',index=False)