import matplotlib.pyplot as plt

import numpy as np

from matplotlib.image import imread

from sklearn.manifold import MDS

from mpl_toolkits import mplot3d

import seaborn as sns

sns.set()

%matplotlib inline
def make_hello(N = 1000, rseed = 42):

    fig, ax = plt.subplots(figsize = (10, 1))

    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top =1)

    ax.axis('off')

    ax.text(0.5, 0.4, "HELLO", va = 'center', ha = 'center', weight = 'bold', size = 85)

    fig.savefig('/kaggle/working/Hello.png')

    plt.close(fig)

    

    data = imread('/kaggle/working/Hello.png')[::-1, :, 0].T

    rng = np.random.RandomState(rseed)

    X = rng.rand(4*N, 2)

    i, j = (X* data.shape).astype(int).T

    

    mask = (data[i, j] < 1)

    X = X[mask]

    X[:, 0] *= (data.shape[0] / data.shape[1])

    X = X[:N]



    return X[np.argsort(X[:, 0])]



X = make_hello(1000)

colorsize = dict(c = X[:, 0], cmap = plt.cm.get_cmap('rainbow', 5))

plt.scatter(X[:, 0], X[:, 1], **colorsize)

plt.axis('equal');

plt.show()
def make_hello(N = 10000, rseed = 42):

    fig, ax = plt.subplots(figsize = (10, 1))

    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top =1)

    ax.axis('off')

    ax.text(0.5, 0.4, "MEJBAH", va = 'center', ha = 'center', weight = 'bold', size = 85)

    fig.savefig('/kaggle/working/mejbah.png')

    plt.close(fig)

    

    data = imread('/kaggle/working/mejbah.png')[::-1, :, 0].T

    rng = np.random.RandomState(rseed)

    X = rng.rand(4*N, 2)

    i, j = (X* data.shape).astype(int).T

    

    mask = (data[i, j] < 1)

    X = X[mask]

    X[:, 0] *= (data.shape[0] / data.shape[1])

    X = X[:N]



    return X[np.argsort(X[:, 0])]



X = make_hello(10000)

colorsize = dict(c = X[:, 0], cmap = plt.cm.get_cmap('rainbow', 3))

plt.scatter(X[:, 0], X[:, 1], **colorsize)

plt.axis('equal');

plt.show()
def make_hello(N = 10000, rseed = 42):

    fig, ax = plt.subplots(figsize = (10, 1))

    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top =1)

    ax.axis('off')

    ax.text(0.5, 0.4, "KAGGLE", va = 'center', ha = 'center', weight = 'bold', size = 85)

    fig.savefig('/kaggle/working/kaggle.png')

    plt.close(fig)

    

    data = imread('/kaggle/working/kaggle.png')[::-1, :, 0].T

    rng = np.random.RandomState(rseed)

    X = rng.rand(4*N, 2)

    i, j = (X* data.shape).astype(int).T

    

    mask = (data[i, j] < 1)

    X = X[mask]

    X[:, 0] *= (data.shape[0] / data.shape[1])

    X = X[:N]



    return X[np.argsort(X[:, 0])]



X = make_hello(10000)

colorsize = dict(c = X[:, 0], cmap = plt.cm.get_cmap('rainbow', 6))

plt.scatter(X[:, 0], X[:, 1], **colorsize)

plt.axis('equal');

plt.show()
def make_hello(N = 10000, rseed = 42):

    fig, ax = plt.subplots(figsize = (10, 1))

    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top =1)

    ax.axis('off')

    ax.text(0.5, 0.4, "GOOGLE", va = 'center', ha = 'center', weight = 'bold', size = 85)

    fig.savefig('/kaggle/working/google.png')

    plt.close(fig)

    

    data = imread('/kaggle/working/google.png')[::-1, :, 0].T

    rng = np.random.RandomState(rseed)

    X = rng.rand(4*N, 2)

    i, j = (X* data.shape).astype(int).T

    

    mask = (data[i, j] < 1)

    X = X[mask]

    X[:, 0] *= (data.shape[0] / data.shape[1])

    X = X[:N]



    return X[np.argsort(X[:, 0])]



X = make_hello(10000)

colorsize = dict(c = X[:, 0], cmap = plt.cm.get_cmap('rainbow', 6))

plt.scatter(X[:, 0], X[:, 1], **colorsize)

plt.axis('equal');

plt.show()
def rotate(X, angle):

    theta = np.deg2rad(angle)

    R = [[np.cos(theta), np.sin(theta)],

        [np.sin(theta), np.cos(theta)]]

    return np.dot(X, R)



X2 = rotate(X, 20) + 5

plt.scatter(X2[:, 0], X2[:, 1], **colorsize)

plt.axis('equal')
def random_projection(X, dimension = 3, rseed= 42):

    assert dimension >= X.shape[1]

    rng = np.random.RandomState(rseed)

    C = rng.randn(dimension, dimension)

    e, V = np.linalg.eigh(np.dot(C, C.T))

    return np.dot(X, V[:X.shape[1]])





X3 = random_projection(X, 3)

X3.shape
plt.figure(figsize = (10, 8))

ax = plt.axes(projection = '3d')

ax.scatter3D(X3[:, 0], X[:, 1], **colorsize)

ax.view_init(azim = 70, elev=50)