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
train = np.load('/kaggle/input/eval-lab-4-f464/train.npy',allow_pickle=True)

test = np.load('/kaggle/input/eval-lab-4-f464/test.npy',allow_pickle=True)
names = []

images = []

for row in train:

    names.append(row[0])

    images.append(row[1])
plt.imshow(images[100], cmap='gray')

plt.show()
#Taking the three red ,green and blue components in our model for the train set.

f_i = []

for image in images:

    f_i.append(image.reshape((1,50*50*3)))
#Taking the three red ,green and blue components in our model for the test set.

t_f_i= []

for row in test:

    t_f_i.append(row[1].reshape((1,50*50*3)))
pd.DataFrame(names)[0].unique()
#Flattening the features vector becuse fit function will only take input less than or equal to 2 dimensions.

f_i = np.array(f_i)

f_i = f_i.reshape((2275,7500))
#Flattening the features vector becuse fit function will only take input less than or equal to 2 dimensions.

t_f_i = np.array(t_f_i)

t_f_i= t_f_i.reshape((976,7500))
import os

from gzip import GzipFile



import numpy as np

import pylab as pl



# from scikit-learn.grid_search import GridSearchCV

# from scikits.learn.metrics import classification_report

# from scikits.learn.metrics import confusion_matrix

# from scikits.learn.pca import RandomizedPCA

# from sklearn.decomposition import RandomizedPCA

# from scikits.learn.svm import SVC

# from sklearn.pca import RandomizedPCA

# from sklearn.decomposition import PCA

from sklearn.decomposition import PCA as RandomizedPCA







n_components = 80

X_train = f_i

# print "Extracting the top %d eigenfaces" % n_components

pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

X_test = t_f_i

# eigenfaces = pca.components_.T.reshape((n_components, 64, 64))



# project the input data on the eigenfaces orthonormal basis

X_tr_pca = pca.transform(X_train)

X_te_pca = pca.transform(X_test)
#Doing gridsearchcv

from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn.svm import SVC



param_grid = {

 'C': [1, 8, 10, 50, 100],

 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01,0.1,'scale'],

}

clf = GridSearchCV(SVC(kernel='rbf'), param_grid)

clf = clf.fit(X_tr_pca, names)

#Predicting the outcomes on the test values.

y_pred = clf.predict(X_te_pca)
out = pd.DataFrame(data={'ImageId':np.arange(0,976),'Celebrity':y_pred})

out.to_csv('submisssss29.csv',index=False)