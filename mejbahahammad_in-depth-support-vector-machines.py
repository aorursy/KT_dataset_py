import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from scipy import stats

from sklearn.datasets.samples_generator import make_blobs

import seaborn as sns

%matplotlib inline
X, y = make_blobs(n_samples=50,

                 centers = 2, 

                 random_state = 0,

                 cluster_std = 0.60)

plt.scatter(X[:, 0],X[:, 1], c = y, s = 50, cmap = plt.cm.autumn)

plt.title('Simple data for classification')

plt.tight_layout()

plt.show()
fit_x = np.linspace(-1, 3.5)

plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = plt.cm.autumn)

plt.plot([0.6], [2.1],'x', color = 'red', markeredgewidth = 2,

        markersize = 10)



for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:

    plt.plot(fit_x, m*fit_x +b,'-k')

    

plt.title('Three perfect Linear discriminative Classifier')

plt.xlim(-1, 3.5)

plt.tight_layout()

plt.show()
from sklearn.svm import SVC



svc_model = SVC(kernel="linear", C = 1e10)

svc_model.fit(X, y)
def plot_svc_decision_function(model, ax =None, plot_support = True):

    if ax is None:

        ax = plt.gca()

    xlim = ax.get_xlim()

    ylim = ax.get_ylim()





    # create grid to evaluatiing model

    x = np.linspace(xlim[0], xlim[1], 30)

    y = np.linspace(ylim[0], ylim[1], 30)

    Y, X = np.meshgrid(y, x)



    xy = np.vstack([X.ravel(), Y.ravel()]).T



    P = model.decision_function(xy).reshape(X.shape)



    ax.contour(X, Y, P, colors = 'K',

              levels = [-1, 0, 1], alpha = 0.5,

              linestyles = ['--','-', '--'])



    if plot_support:

        ax.scatter(model.support_vectors_[:, 0],

                  model.support_vectors_[:, 1],

                  s = 300, linewidth = 1,

                  facecolors = 'none');



    ax.set_xlim(xlim)

    ax.set_ylim(ylim)

plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = plt.cm.autumn)

plot_svc_decision_function(svc_model)
svc_model.support_vectors_
def plot_SVM(N = 10, ax = None):

    X, y = make_blobs(n_samples=200, centers=2,

                     random_state = 0, cluster_std = 0.60)

    X = X[:N]

    y = y[:N]

    

    

    model = SVC(kernel='linear', C = 1e10)

    model.fit(X, y)

    

    ax = ax or plt.gca()

    

    ax.scatter(X[:, 0], X[:, 1], c = y, s= 50, cmap = plt.cm.autumn)

    ax.set_xlim(-1, 4)

    ax.set_ylim(-1, 6)

    plot_svc_decision_function(model, ax)
fig,ax = plt.subplots(1, 2, figsize = (16, 6))

fig.subplots_adjust(left = 0.0625, right = 0.95, wspace = 0.1)

for axi, N in zip(ax, [60, 120]):

    plot_SVM(N, axi)

    axi.set_title('N = {0}'.format(N))
from ipywidgets import interact, fixed

interact(plot_SVM, N = [10,30, 40, 200], ax = fixed(None));