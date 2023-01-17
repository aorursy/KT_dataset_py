import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
%matplotlib inline
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
national_data = pd.read_csv("../input/national_data.csv", index_col='Region')
avg_gdp = pd.read_csv("../input/avg_gdp.csv", index_col='Region')
# national_data['Region'].astype(str)
national_data['Avg_GDP'] = avg_gdp['Avg_GDP']
national_data
national_data.describe()
national_data_norm = national_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
national_data_norm
X = national_data_norm.values
print(X)
plt.scatter(X[:, 1], X[:, 2])
plt.scatter(X[:, 1], X[:, 4])
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X[:, 1], X[:, 3])
plt.scatter(X[:, 1], X[:, 5])
X.shape
# generate the linkage matrix
Z = linkage(X, 'ward')
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X))
c
Z[0]
Z[1]
Z
idxs = [13, 19]
plt.figure(figsize=(8, 6))
plt.scatter(X[:,1], X[:,5])  # plot all points
plt.scatter(X[idxs,1], X[idxs,5], c='r')  # plot interesting points in red again
plt.show()
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('cities')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.show()
# set cut-off to 1.5
# 从1.5的距离处断开
max_d = 1.5  # max_d as in max_distance
fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()
from scipy.cluster.hierarchy import fcluster
max_d = 1.5
clusters = fcluster(Z, max_d, criterion='distance')
clusters
national_data['Class'] = clusters
national_data[national_data['Class'] == 3]
class_1 = national_data[national_data['Class'] == 1]
class_2 = national_data[national_data['Class'] == 2]
class_3 = national_data[national_data['Class'] == 3]
class_1
class_2
class_3