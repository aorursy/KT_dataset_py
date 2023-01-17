from sklearn.datasets import make_classification 

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

np.random.seed(2019)



# generate dataset 

X, y = make_classification(1024, 2, n_informative=2, n_clusters_per_class=1, n_redundant=0, flip_y=0.05, random_state=125)





plt.scatter(X[y==0, 0], X[y==0, 1], label='target=0')

plt.scatter(X[y==1, 0], X[y==1, 1], label='target=1')

plt.legend()
from sklearn.datasets import make_classification 

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

np.random.seed(2019)



# generate dataset 

X, y = make_classification(1024, 2, n_informative=2, n_clusters_per_class=2, n_redundant=0, flip_y=0.05, random_state=5)





plt.scatter(X[y==0, 0], X[y==0, 1], label='target=0')

plt.scatter(X[y==1, 0], X[y==1, 1], label='target=1')

plt.legend()
from sklearn.datasets import make_classification 

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



# generate dataset 

X, y = make_classification(1024, 2, n_informative=2, n_clusters_per_class=2, n_redundant=0, flip_y=0.05, random_state=0)





plt.scatter(X[y==0, 0], X[y==0, 1], label='target=0')

plt.scatter(X[y==1, 0], X[y==1, 1], label='target=1')

plt.legend()
from sklearn import mixture

from matplotlib.colors import LogNorm

from sklearn.covariance import OAS



def get_mean_cov(X):

    model = OAS(assume_centered=False)

    

    ms = []

    ps = []

    for xi in X:

        model.fit(xi)

        ms.append(model.location_)

        ps.append(model.precision_)

    return np.array(ms), np.array(ps)



knn_clf = mixture.GaussianMixture(n_components=2, init_params='random',

                          covariance_type='full',

                          n_init=1, 

                          random_state=0)





X, y = make_classification(1024, 2, n_informative=2, n_clusters_per_class=2, n_redundant=0, flip_y=0.05, random_state=5)



x_t =X.copy()

y_t =y.copy()

train3_pos = X[y==1]

train3_neg = X[y==0]



print(train3_pos.shape, train3_neg.shape)

ms, ps = get_mean_cov([train3_pos, train3_neg])



clf = mixture.GaussianMixture(n_components=2, covariance_type='full', means_init=ms, precisions_init=ps,)

clf.fit(X)



x = np.linspace(-5., 5.)

y = np.linspace(-5., 5.)

X, Y = np.meshgrid(x, y)

XX = np.array([X.ravel(), Y.ravel()]).T

Z = -clf.score_samples(XX)

Z = Z.reshape(X.shape)



plt.figure()

plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),  levels=np.logspace(0, 3, 10))

plt.scatter(x_t[y_t==0, 0], x_t[y_t==0, 1], label='target=0', alpha=.3)

plt.scatter(x_t[y_t==1, 0], x_t[y_t==1, 1], label='target=1', alpha=.3)

plt.legend()
from sklearn import mixture

from matplotlib.colors import LogNorm

from sklearn.covariance import OAS

X, y = make_classification(1024, 2, n_informative=2, n_clusters_per_class=2, n_redundant=0, flip_y=0.05, random_state=5)



x_t =X.copy()

y_t =y.copy()

train3_pos = X[y==1]

train3_neg = X[y==0]



cluster_num_pos = knn_clf.fit_predict(train3_pos)

train3_pos_1 = train3_pos[cluster_num_pos==0]

train3_pos_2 = train3_pos[cluster_num_pos==1]

#print(train3_pos.shape, train3_pos_1.shape, train3_pos_2.shape, train3_pos_3.shape)



## FIND CLUSTERS IN CHUNKS WITH TARGET = 0

cluster_num_neg = knn_clf.fit_predict(train3_neg)

train3_neg_1 = train3_neg[cluster_num_neg==0]

train3_neg_2 = train3_neg[cluster_num_neg==1]

        

    

print(train3_pos.shape, train3_neg.shape)

ms, ps = get_mean_cov([train3_pos_1, train3_pos_2, train3_neg_1, train3_neg_2])



clf = mixture.GaussianMixture(n_components=4, covariance_type='full', means_init=ms, precisions_init=ps,)

clf.fit(X)



x = np.linspace(-5., 5.)

y = np.linspace(-5., 5.)

X, Y = np.meshgrid(x, y)

XX = np.array([X.ravel(), Y.ravel()]).T

Z = -clf.score_samples(XX)

Z = Z.reshape(X.shape)



plt.figure()

plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),  levels=np.logspace(0, 3, 10))

plt.scatter(x_t[y_t==0, 0], x_t[y_t==0, 1], label='target=0', alpha=.3)

plt.scatter(x_t[y_t==1, 0], x_t[y_t==1, 1], label='target=1', alpha=.3)

plt.legend()