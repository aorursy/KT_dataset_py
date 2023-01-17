import pandas as pd

import hypertools as hyp 

%matplotlib inline
data = pd.read_csv('../input/mushrooms.csv')

data.head()
class_labels = data.pop('class')
fig, ax, data, _ = hyp.plot(data,'.') # if the number of features is greater than 3, the default is to plot in 3d
fig, ax, data, _ = hyp.plot(data,'.', group=class_labels, legend=list(set(class_labels)))
fig, ax, data, _ = hyp.plot(data, '.', n_clusters=23)



# you can also recover the cluster labels using the cluster tool

cluster_labels = hyp.tools.cluster(data, n_clusters=23, ndims=3) 

# hyp.plot(data, 'o', point_colors=cluster_labels, ndims=2)
fig, ax, data, _ = hyp.plot(data,'.', group=cluster_labels, palette="deep")
fig, ax, data, _ = hyp.plot(data,'.', model='FastICA', group=class_labels, legend=list(set(class_labels)))
fig, ax, data, _ = hyp.plot(data, '.', model='TSNE', group=class_labels, legend=list(set(class_labels)))