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
import h5py 

import matplotlib.pyplot as plt #import plotting modules

import seaborn as sns

%matplotlib inline
with h5py.File('../input/all_mias_scans.h5', 'r') as scan_h5:

    bg_info = scan_h5['BG'][:]

    class_info = scan_h5['CLASS'][:]

    # low res scans

    scan_lr = scan_h5['scan'][:][:, ::16, ::16]
scan_lr.shape
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()

class_le.fit(class_info)

class_vec = class_le.transform(class_info)

class_le.classes_
class_le.classes_
from sklearn.feature_extraction import image

deneme=image.extract_patches_2d(scan_lr[0],(3,3))

deneme
plt.imshow(scan_lr[14],cmap="gray")
scan_lr[14].shape
set(deneme.flatten())
class_info.shape
from skimage.feature import local_binary_pattern

lbp = local_binary_pattern(scan_lr[14], 16, 4, "uniform")
set(lbp.flatten())
lbp.shape
lbp.ravel().shape
set(lbp.ravel())
sns.countplot(x=lbp.ravel())
features=[]

for img in scan_lr:

    lbp = local_binary_pattern(img, 16, 4, "uniform")

    flatten=lbp.ravel()

    features.append(flatten)

    
x=pd.DataFrame(features)

x.head(5)
len(x)
from sklearn.decomposition import PCA

pca=PCA(n_components=10)

transx=pca.fit_transform(x)
transx.shape
X=transx

y=class_info
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1945)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

scalex=sc.fit_transform(X)
scalex=pd.DataFrame(scalex)

scalex.head(5)
Xs_train, Xs_test, y_train, y_test = train_test_split(scalex, y, test_size=0.20, random_state=1945)

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

tuned_parameters = [{'kernel': ['rbf'], 'gamma': ["auto"],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear',"sigmoid"], 'C': [1, 10, 100, 1000]}

                   ]

scores ='accuracy'

clf = GridSearchCV(SVC(), tuned_parameters, cv=5,

                       scoring='accuracy')

clf.fit(Xs_train, y_train)
print(clf.best_params_)
from sklearn.svm import SVC



svm=SVC(kernel="rbf",gamma="auto")

svm.fit(Xs_train,y_train)

ypred=svm.predict(Xs_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.neighbors import KNeighborsClassifier

tuned_parameters = [

                     {"n_neighbors":[1,3,5,7,9,11]}

                   ]

clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5,

                       scoring='accuracy')

clf.fit(Xs_train, y_train)

print(clf.best_params_)
svm=KNeighborsClassifier(n_neighbors=9)

svm.fit(Xs_train,y_train)

ypred=svm.predict(Xs_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
svm=KNeighborsClassifier(n_neighbors=3)

svm.fit(Xs_train,y_train)

ypred=svm.predict(Xs_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
svm=KNeighborsClassifier(n_neighbors=1)

svm.fit(Xs_train,y_train)

ypred=svm.predict(Xs_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from skimage.feature import peak_local_max

peak_local_max(scan_lr[4],min_distance=4).ravel()
localmax=[]

for img in scan_lr:

    features=peak_local_max(img,min_distance=4)

    localmax.append(features.ravel())

x2=pd.DataFrame(localmax)

x2.head()
x2.isnull().sum()