import pandas as pd
import numpy as np

X_train = pd.read_csv('../input/train.csv').values[:,1:]  # exclude the target
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)  # rescale to (0, 1)
from sklearn.neural_network import BernoulliRBM
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, random_state=42, verbose=True)
rbm.fit(X_train)
def gen_mnist_image(X):
    return np.rollaxis(np.rollaxis(X[0:200].reshape(20, -1, 28, 28), 0, 2), 1, 3).reshape(-1, 20 * 28)

xx = X_train[:40].copy()
for _ in range(1000):
    for n in range(40):
        xx[n] = rbm.gibbs(xx[n])
        
import matplotlib.pyplot as plt
plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(xx), cmap='gray')
plt.figure(figsize=(20, 20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.RdBu,
               interpolation='nearest', vmin=-2.5, vmax=2.5)
    plt.axis('off')