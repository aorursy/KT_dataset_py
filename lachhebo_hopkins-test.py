import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.decomposition import PCA

from sklearn import datasets

from sklearn.preprocessing import scale

from pyclustertend import hopkins ## the hopkins test





from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt







heart_df = pd.read_csv("../input/heart.csv")
X = heart_df[heart_df.columns[~heart_df.columns.isin(["target"])]].values

y = heart_df[heart_df.columns[heart_df.columns.isin(["target"])]].values.flatten()
hopkins(X, X.shape[0])
hopkins(scale(X),X.shape[0])
pca = PCA(n_components = 2)



X_pca = pca.fit_transform(scale(X))



labels = heart_df.target.values



cdict = {0 : "green", 1 : "red"}

labl = {0: "healthy", 1 : "sick"}

marker = {0 : "o", 1: "*"}

alpha = {0: 0.5, 1: 0.5}



#fig = plt.figure(figsize=(10, 10))

#ax = fig.add_subplot(111, projection='3d')



fig = plt.figure(figsize=(10, 10))





for l in np.unique(labels):

    ix = np.where(labels == l)

    plt.scatter(X_pca[ix,0],X_pca[ix,1], c = cdict[l], s=40, label = labl[l], marker = marker[l], alpha = alpha[l])



#plt.scatter(X_pca[:,0],X_pca[:,1]);

plt.xlabel("first principal component")

plt.xlabel("second principal component")

plt.title("PCA  : heart diseases")