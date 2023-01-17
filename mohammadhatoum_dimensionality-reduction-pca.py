import warnings

warnings.filterwarnings("ignore")

import numpy as np

# Load the diabetes dataset

import pandas as pd



import matplotlib.pyplot as plt   

from skimage.io import imshow



#System

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

print("Warnings ignored!!")
data=np.load("../input/olivetti_faces.npy")

labels=np.load("../input/olivetti_faces_target.npy")



print(f"Shape of inputs: {data.shape}")

print(f"Shape of labels: {labels.shape}")

print(f"Unique values for labels: {np.unique(labels)}")
imshow(data[0]) 
X=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))

print("After reshape:",X.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, labels, test_size=0.25, stratify=labels, random_state=0)

print("X_train shape:",X_train.shape)

print("y_train shape:{}".format(y_train.shape))
from sklearn.decomposition import PCA

pca=PCA()

pca.fit(X)



plt.figure(1, figsize=(12,8))



plt.plot(pca.explained_variance_, linewidth=2)

 

plt.xlabel('Components')

plt.ylabel('Explained Variaces')

plt.show()
n_components=60

pca=PCA(n_components=n_components, whiten=True)

pca.fit(X)
fig,ax=plt.subplots(1,1,figsize=(8,8))

ax.imshow(pca.mean_.reshape((64,64)), cmap="gray")

ax.set_xticks([])

ax.set_yticks([])

ax.set_title('Average Face')
number_of_eigenfaces=len(pca.components_)

eigen_faces=pca.components_.reshape((number_of_eigenfaces, data.shape[1], data.shape[2]))



cols=10

rows=int(number_of_eigenfaces/cols)

fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))

axarr=axarr.flatten()

for i in range(number_of_eigenfaces):

    axarr[i].imshow(eigen_faces[i],cmap="gray")

    axarr[i].set_xticks([])

    axarr[i].set_yticks([])

    axarr[i].set_title("eigen id:{}".format(i))

plt.suptitle("All Eigen Faces".format(10*"=", 10*"="))
X_train_pca=pca.transform(X_train)

X_test_pca=pca.transform(X_test)

print(f"Shape before {X_train.shape} vs shape after {X_train_pca.shape}")
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



clf = LogisticRegression()

clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)



print("Accuracy score:{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))