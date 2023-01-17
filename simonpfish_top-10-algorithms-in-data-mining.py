import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz 

####### play around with these #####
## Play around with these, you can find the explanation in the docs
criterion = 'gini'
splitter = 'best'
max_depth = 50
min_samples_split = 2
min_samples_leaf = 1
min_weight_fraction_leaf = 0
max_features = None
max_leaf_nodes = 10
####################################

iris = load_iris()

clf = tree.DecisionTreeClassifier(criterion = criterion ,splitter = splitter,
                                  max_depth = max_depth, min_samples_split = min_samples_split,
                                  min_samples_leaf = min_samples_leaf, 
                                  min_weight_fraction_leaf = min_weight_fraction_leaf, 
                                  max_features = max_features, max_leaf_nodes = max_leaf_nodes)

clf = clf.fit(iris.data, iris.target)

## plotting code, don't worry about this:

n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Plot the decision boundary
    clf = clf.fit(X, y)
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()


clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 

dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,  
                                 class_names=iris.target_names,  
                                 filled=True, rounded=True,  
                                 special_characters=True)

graph = graphviz.Source(dot_data)  
graph
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

####### play around with these #####
# data:
n_samples = 500
n_clusters = 4
cluster_std = 1

# model:
n_centroids = 4
init = 'random' # 'k-means++'
n_init = 10
max_iter = 100
####################################

X, y = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=cluster_std) 
kmeans = KMeans(n_clusters, init=init, n_init=n_init, max_iter=max_iter)
y_pred = kmeans.fit_predict(X)

plt.style.use('seaborn-pastel')
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
X = iris.data[iris.target != 0, :2]
y = iris.target[iris.target != 0]

n_sample = len(X)

X_train = X[:int(.9 * n_sample)]
y_train = y[:int(.9 * n_sample)]
X_test = X[int(.9 * n_sample):]
y_test = y[int(.9 * n_sample):]

####### play around with these #####
C = 1.0  # SVM regularization parameter
kernel = 'rbf' #  ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
gamma = 5  # for ‘rbf’, ‘poly’ and ‘sigmoid’
degree = 3  # for ‘poly’
####################################

clf = svm.SVC(kernel=kernel, gamma=gamma, C=C)
clf.fit(X_train, y_train)

plt.axis('tight')
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)

plt.clf()
plt.style.use('seaborn-dark')
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,edgecolor='k', s=20)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.show()
# code adapted from:
#  http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.12-Gaussian-Mixtures.ipynb

from matplotlib.patches import Ellipse 
from sklearn.datasets.samples_generator import make_blobs
import sklearn.mixture as mix
import matplotlib.pyplot as plt
import numpy as np

####### play around with these #####
# data:
n_clusters = 6
n_samples = 400
cluster_std = .6

# model:
n_components = 4 
max_iter = 500
tol = 1
covariance_type = 'full' # {‘tied’, ‘diag’, ‘spherical’}
n_init = 1
####################################

X, y_true = make_blobs(n_samples = n_samples,
                       centers = n_clusters,
                       cluster_std = cluster_std)

gmm = mix.GaussianMixture(n_components=n_components,
                          max_iter=max_iter, 
                          tol=tol, 
                          covariance_type =covariance_type, 
                          n_init=n_init)

Y_pred = gmm.fit(X).predict(X)

# plotting code. don't worry about this:

dot_size = 50
cmap = 'viridis'

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
        
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, 
                            angle, **kwargs))
        
def plot_gmm(gmm, labels, label=True, ax=None):
    
    fig, ax = plt.subplots(figsize=(9,7))      
    ax = ax or plt.gca()

    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=dot_size, cmap=cmap, zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=dot_size, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, ax=ax, alpha=w * w_factor)
        
plot_gmm(gmm, Y_pred)

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

####### play around with these #####
alpha = .5 # damping factor which defaults to .85
max_iter = 100

## Feel free to add to the graph bellow
####################################

G = nx.DiGraph()
G.add_edges_from(
    [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), 
     ('E', 'F'),('B', 'H'), ('B', 'G'), ('B', 'F'), 
     ('C', 'G'), ('E', 'G'), ('A', 'H'), ('A', 'I')]
)

nx.draw_kamada_kawai(G, with_labels=True, node_color = 'red', node_size = 1500)
plt.show()

pr = nx.pagerank(G, alpha=alpha, max_iter=max_iter, )

# sorted print:
for key, value in sorted(pr.items(), key=lambda x: 1/x[1]): 
    print("{} : {}".format(key, value))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

####### play around with these #####
# data:
randomness = 0.25
n_samples = 100

# model:
weak_learner = DecisionTreeRegressor(max_depth=4)
n_estimators = 300
learning_rate = 1.0
loss = 'linear' # 'linear', 'square', 'exponential'
####################################

# Create the dataset
X = np.linspace(0, 6, 200)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.normal(0, randomness, X.shape[0])

# Fit regression model
regr = AdaBoostRegressor(weak_learner, 
                         n_estimators=n_estimators, 
                         learning_rate=learning_rate)
regr.fit(X, y)

# Predict
y = regr.predict(X)
y_1 = np.sin(X).ravel() + np.sin(6 * X).ravel()

# Plot the results
plt.figure()
plt.style.use('seaborn-dark')
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y, c="r", label="estimate", linewidth=2)
plt.plot(X, y_1, c="g", label="truth", linewidth=2)

plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()

plt.show()