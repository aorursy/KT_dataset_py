import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import make_blobs


data, _ = make_blobs(n_samples=1000, centers=[(-10,10), (10,-10), (10,10), (-10,-10)], n_features=2,

                  random_state=0)

print('data.shape :', data.shape)

plt.plot(data[:,0], data[:,1], '.')
W = np.array([[20,20,-20,-20], [20,-20,20,-20]], dtype = 'float')

# W.shape

W = W.T

W.shape

plt.scatter(W[:,0], W[:,1])
lr = 0.001

for iter in range(10):

    for i in range(data.shape[0]):

        edist = np.linalg.norm(W - data[i,:], axis=1)

        ind = np.argmin(edist)

        W[ind,:] = (1 - lr)*W[ind,:] + lr*data[i,:]

    
plt.scatter(W[:,0], W[:,1])
print("updated weights :\n", W)