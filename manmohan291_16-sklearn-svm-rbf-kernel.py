import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import itertools

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
np.random.seed(0) 

X = np.random.randn(200, 2)

whereClause = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0) 

Y = np.where(whereClause, 1, -1)

Y.shape=(len(Y),1)

cmap = ListedColormap(['blue', 'red']) 

cmaplight = ListedColormap(['cyan', 'orange']) 

plt.scatter(X[:, 0],X[:, 1], c=Y[:,0],marker='x', cmap=cmap)

plt.show() 
def plot_Decision_Boundry(X,Y,clf):   

   

    

    plt.figure(figsize=(8,8))

    

    #Predict for each X1 and X2 in Grid 

    x_min, x_max = X[:, 0].min() , X[:, 0].max() 

    y_min, y_max = X[:, 1].min() , X[:, 1].max() 

    u = np.linspace(x_min, x_max, 100) 

    v = np.linspace(y_min, y_max, 100) 



    U,V=np.meshgrid(u,v)

    UV=np.column_stack((U.flatten(),V.flatten())) 

    W=clf.predict(UV) 

    W.shape=U.shape

    plt.contourf(U, V,  W, cmap=cmaplight, alpha=0.4)

   

    plt.scatter(X[:,0],X[:,1], c=Y[:,0],marker='.', cmap=cmap) 

 



    

    plt.show()
clf = SVC(kernel='rbf',gamma=1,C=0.1)

clf.fit(X,Y.flatten())

y_pred = clf.predict(X)

print("Accuracy=",accuracy_score(Y.flatten(),y_pred))

plot_Decision_Boundry(X,Y,clf)
clf = SVC(kernel='rbf',gamma=1,C=1000)

clf.fit(X,Y.flatten())

y_pred = clf.predict(X)

print("Accuracy=",accuracy_score(Y.flatten(),y_pred))

plot_Decision_Boundry(X,Y,clf)
clf = SVC(kernel='rbf',gamma=10,C=0.1)

clf.fit(X,Y.flatten())

y_pred = clf.predict(X)

print("Accuracy=",accuracy_score(Y.flatten(),y_pred))

plot_Decision_Boundry(X,Y,clf)
clf = SVC(kernel='rbf',gamma=0.1,C=0.1)

clf.fit(X,Y.flatten())

y_pred = clf.predict(X)

print("Accuracy=",accuracy_score(Y.flatten(),y_pred))

plot_Decision_Boundry(X,Y,clf)