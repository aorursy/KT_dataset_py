from time import time
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
from sklearn import datasets

n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)


fig = plt.figure()
ax = p3.Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)
plt.show()
def calcDistance(X):
    m,n = X.shape
    distance = np.zeros((m,m))
    for i in range(m):
        for j in range(i+1,m):
            distance[i,j] = np.linalg.norm(X[i,:]-X[j,:])
            distance[j,i] = distance[i,j]
    return distance
def mds(D,q):
    D = np.asarray(D)
    DSquare = D**2
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis = 0)
    rowMean = np.mean(DSquare, axis = 1)
    B = np.zeros(DSquare.shape)
    
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5*(DSquare[i][j] - rowMean[i] - columnMean[j]+totalMean)
    eigVal,eigVec = np.linalg.eig(B)
    X = np.dot(eigVec[:,:q],np.sqrt(np.diag(eigVal[:q])))

    return X.astype(np.float64)
distance = calcDistance(X)
new_X = mds(distance,2)
plt.plot(new_X[:, 0], new_X[:, 1],'x')
distance = calcDistance(X)
new_X = mds(distance,3)

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2])
# ax.view_init(4, -72)
plt.show()

