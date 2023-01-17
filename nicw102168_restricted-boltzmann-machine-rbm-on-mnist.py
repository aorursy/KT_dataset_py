import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline   

plt.rcParams['image.cmap'] = 'gray'



import pandas as pd
def gen_mnist_image(X):

    return np.rollaxis(np.rollaxis(X[0:200].reshape(20, -1, 28, 28), 0, 2), 1, 3).reshape(-1, 20 * 28)
X_train = pd.read_csv('../input/train.csv').values[:,1:]

X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)  # 0-1 scaling
plt.figure(figsize=(10,20))

plt.imshow(gen_mnist_image(X_train));
from sklearn.neural_network import BernoulliRBM

rbm = BernoulliRBM(n_components=100, learning_rate=0.01, random_state=0, verbose=True)

rbm.fit(X_train)
xx = X_train[:40].copy()

for ii in range(1000):

    for n in range(40):

        xx[n] = rbm.gibbs(xx[n])
plt.figure(figsize=(10,20))

plt.imshow(gen_mnist_image(xx))
xx = X_train[:40].copy()

for ii in range(10000):

    for n in range(40):

        xx[n] = rbm.gibbs(xx[n])
plt.figure(figsize=(10,20))

plt.imshow(gen_mnist_image(xx))
plt.figure(figsize=(20, 20))

for i, comp in enumerate(rbm.components_):

    plt.subplot(10, 10, i + 1)

    plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.RdBu,

               interpolation='nearest', vmin=-2.5, vmax=2.5)

    plt.axis('off')

plt.suptitle('100 components extracted by RBM', fontsize=16);