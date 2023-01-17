import numpy as np

from sklearn.svm import SVC

from sklearn.datasets.samples_generator import make_blobs

from sklearn.datasets.samples_generator import make_circles

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d
samples, labels = make_blobs(n_samples=200, centers=2,

                  random_state=0, cluster_std=0.50)
plt.scatter(samples[:, 0], samples[:, 1], c=labels);
clf = SVC(kernel='linear')

clf.fit(samples, labels)
def plot_svc_decision_function(clf, ax=None, plot_support=True):

    """Plot the decision function for a 2D SVC"""

    if ax is None:

        ax = plt.gca()

    xlim = ax.get_xlim()

    ylim = ax.get_ylim()



    # create grid to evaluate model

    x = np.linspace(xlim[0], xlim[1], 2000)

    y = np.linspace(ylim[0], ylim[1], 2000)

    Y, X = np.meshgrid(y, x)

    xy = np.vstack([X.ravel(), Y.ravel()]).T

    P = clf.predict(xy).reshape(X.shape)



    # plot decision boundary and margins

    ax.contourf(X, Y, P, levels=2, alpha=0.2)

    

    ax.set_xlim(xlim)

    ax.set_ylim(ylim)
plt.scatter(samples[:, 0], samples[:, 1], c=labels)

plot_svc_decision_function(clf);
samples, labels = make_circles(200, factor=0.1, noise=0.1)
plt.scatter(samples[:, 0], samples[:, 1], c=labels);
clf.fit(samples, labels)
plt.scatter(samples[:, 0], samples[:, 1], c=labels)

plot_svc_decision_function(clf);
rbf_kernel_clf = SVC(kernel='rbf')

rbf_kernel_clf.fit(samples, labels)
plt.scatter(samples[:, 0], samples[:, 1], c=labels)

plot_svc_decision_function(rbf_kernel_clf);
r = np.exp(-(samples ** 2).sum(1))
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

ax.plot(samples[:,0], samples[:,1], r, 'o', markersize=8, alpha=0.5)

plt.show()