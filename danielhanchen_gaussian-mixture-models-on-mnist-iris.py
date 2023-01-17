from numpy import *

from pandas import *

import matplotlib.pyplot as plt

%matplotlib inline



data = read_csv("../input/iris.data.csv", header = None)



data[4] = data[4].astype("category")

data[4] = data[4].cat.codes



Y = array(data.pop(4))

X = array(data)
from sklearn.mixture import GaussianMixture as mix

model = mix(n_components = len(np.unique(Y)))
model.fit(X).predict(X)
from sklearn import datasets as data

mnist = data.load_digits()



plt.gray()

plt.matshow(mnist.images[100])

plt.show()



Y = mnist.target

X = mnist.images



X = X.reshape(len(X),-1)
from sklearn.mixture import GaussianMixture as mix

model = mix(n_components = 10, init_params='kmeans',

           n_init = 5, max_iter = 5000, covariance_type = 'diag')

model.fit(X)



preds = model.predict(X)

labels = {}

seen = []



for dist in range(10):

    part = Y[where(preds==dist)]

    print(part)
most
preds = np.array([labels[x] for x in preds])
sum(preds==Y)/len(Y)