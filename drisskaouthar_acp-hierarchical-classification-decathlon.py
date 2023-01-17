

import pandas as pd

my_data = pd.read_csv('../input/decathlon.csv')



        
import pandas as pd 
data_decathlon= pd.read_csv('../input/decathlon.csv', sep=',',nrows=41)
data_decathlon.set_index('Unnamed: 0')
print (data_decathlon)

type(data_decathlon)
data_decathlon.shape


from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

import pandas, numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=4)
new=data_decathlon.ix[:,[1,2,3,4,5,6,7,8,9,10]]
print(new)

#The first axis explains the most of the variance(we have 4 principal axes )
print(pca.fit(new))
pca.explained_variance_ratio_
plt.bar(numpy.arange(len(pca.explained_variance_ratio_))+0.5, pca.explained_variance_ratio_)
plt.title("Barplotof the explaned variance");
X_reduced = pca.transform(new)
plt.figure(figsize=(18,6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])

for label, x, y in zip(new.index, X_reduced[:, 0], X_reduced[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (-10, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.title("ACP approximating similar evolution patterns");
#Hierarchical ascending classification
from sklearn.cluster import AgglomerativeClustering
ward = AgglomerativeClustering(linkage='ward', compute_full_tree=True).fit(new)
dendro = [ ]
for a,b in ward.children_:
    dendro.append([a,b,float(len(dendro)+1),len(dendro)+1])

from scipy.cluster.hierarchy import dendrogram
fig = plt.figure( figsize=(8,8) )
ax = fig.add_subplot(1,1,1)

r = dendrogram(dendro, color_threshold=1, labels=new.index, show_leaf_counts=True, ax=ax, orientation = "left")