from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
scan_lr[24].shape
class_info.shape
from skimage.feature import hog

his=hog(scan_lr, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),  transform_sqrt=False, feature_vector=True)
set(his.flatten())
his.shape
his.ravel().shape
set(his.ravel())
sns.countplot(x=his.ravel())
features=[]

for img in scan_lr:

    his = hog(scan_lr, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),  transform_sqrt=False, feature_vector=True)

    flatten=his.ravel()

    features.append(flatten)

    
x=pd.DataFrame(features)



x=x.fillna(0)

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
from sklearn.svm import SVC



svm=SVC(kernel="rbf",gamma="auto")

svm.fit(X_train,y_train)

ypred=svm.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train,y_train)

ypred=knn.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 9)

knn.fit(X_train,y_train)

ypred=knn.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 12)

knn.fit(X_train,y_train)

ypred=knn.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 15)

knn.fit(X_train,y_train)

ypred=knn.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 18)

knn.fit(X_train,y_train)

ypred=knn.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)

knn.fit(X_train,y_train)

ypred=knn.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from xgboost import XGBClassifier

xgc = XGBClassifier(silent = False, nthread=2)

xgc.fit(X_train, y_train)

y_pred = xgc.predict(X_test)

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

nn.fit(X_train, y_train)

y_pred = nn.predict(X_test)

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.tree import DecisionTreeClassifier

tr = DecisionTreeClassifier(random_state=0)

tr.fit(X_train, y_train)

y_pred = tr.predict(X_test)

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
from sklearn.naive_bayes import GaussianNB

gau = GaussianNB()

gau.fit(X_train, y_train)

y_pred = gau.predict(X_test)

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))