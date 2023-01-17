import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
# Feel free to play with these parameters for the tree.
#criterion : string, optional (default=”gini”) can also try "entropy"
# splitter : string, optional (default=”best”) or “random” to choose the best random split.
# max_depth : int or None, optional (default=None) If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# min_samples_split : int, float, optional (default=2) The minimum number of samples required to split an internal node
# min_samples_leaf : int, float, optional (default=1) The minimum number of samples required to be at a leaf node. 
# min_weight_fraction_leaf : float, optional (default=0.)The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. 
# max_features : int, float, string or None, optional (default=None) The number of features to consider when looking for the best split:
#random_state dont worry about this one just leave blank
# max_leaf_nodes : int or None, optional (default=None)
######################################
## Play around with these, you can find the explanation above
criteria = 'gini'
split = 'best'
depth = None
min_split = 2
min_samples = 1
min_weight = 0
features = None
max_leaves = None
#######################################################################################################################################################
clf = tree.DecisionTreeClassifier(criterion = criteria ,splitter = split,max_depth = depth,min_samples_split = min_split,min_samples_leaf = min_samples, min_weight_fraction_leaf = min_weight, max_features = features, max_leaf_nodes = max_leaves)
clf = clf.fit(iris.data, iris.target)

import graphviz 
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

# code sourced from:
#  http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.12-Gaussian-Mixtures.ipynb
from matplotlib.patches import Ellipse 
from sklearn.datasets.samples_generator import make_blobs
import sklearn.mixture as mix
import matplotlib.pyplot as plt
import numpy as np

#########################################################
# Adjust the K up here to make more clusters in the data
k = 6
#########################################################
n_draws = 500
sigma = .7
random_state = 0
dot_size = 50
cmap = 'viridis'
X3, y_true = make_blobs(n_samples = 400,
                       centers = k,
                       cluster_std = .6,
                       random_state = random_state)
X3 = X3[:, ::-1] # better plotting

fig, ax = plt.subplots(figsize=(9,7))
ax.scatter(X3[:, 0], X3[:, 1], s=dot_size)
plt.title('Blobs Prior to grouping', fontsize=18, fontweight='demi')

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
        
def plot_gmm(gmm, X, label=True, ax=None):
    
    fig, ax = plt.subplots(figsize=(9,7))      
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=dot_size, cmap=cmap, zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=dot_size, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, ax=ax, alpha=w * w_factor)
# n_components : int, defaults to 1.The number of mixture components.
# tol : float, defaults to 1e-3.The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
# max_iter : int, defaults to 100. The number of EM iterations to perform.
##################################
## Play with these
num_components = 4
iterations = 500
########################################################################################################################################
gmm = mix.GaussianMixture(n_components = num_components, max_iter = iterations, random_state=random_state)
plot_gmm(gmm, X3)
plt.title('Clusters after grouping', fontsize=18, fontweight='demi')
rng = np.random.RandomState(13)
X3_stretched = np.dot(X3, rng.randn(2, 2))

# lets test on the stretched data set
##################################
## Play with these
num_components = 4
iterations = 500
########################################################################################################################################
gmm = mix.GaussianMixture(n_components=num_components, max_iter = iterations, random_state=random_state+1)
plot_gmm(gmm, X3_stretched)
plt.title('Clusters about grouping after stretching', fontsize=18, fontweight='demi')

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

##############################
## Feel free to add to the graph
##############################
G = nx.DiGraph()
G.add_edges_from(
    [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
     ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G'), ('E', 'G'), ('A', 'H'), ('A', 'I')])

nx.draw(G, with_labels=True, node_color = 'red', node_size = 1500)
plt.show()
#########################
# The alpha value is a damping factor which defaults to .85
a = .5
###########################
pr = nx.pagerank(G, alpha=a)
print(pr)
