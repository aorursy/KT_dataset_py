%%time

# We add the rapids kaggle dataset [Link](https://www.kaggle.com/cdeotte/rapids)

# This installs the package offline. Installation takes place under a minute! 

import sys

!cp ../input/rapids/rapids.0.11.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf,cuml

import pandas as pd

import numpy as np

from cuml.manifold import TSNE

from cuml import PCA  

#from cuml.decomposition import PCA << this is also supported

from cuml.cluster import DBSCAN

#from cuml import DBSCAN << this is also supported

import matplotlib.pyplot as plt

%matplotlib inline
def scatter_thumbnails(data, images, zoom=0.12, colors=None):

    assert len(data) == len(images)



    # reduce embedding dimentions to 2

    x = PCA(n_components=2).fit_transform(data) if len(data[0]) > 2 else data



    # create a scatter plot.

    f = plt.figure(figsize=(22, 15))

    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], s=4)

    _ = ax.axis('off')

    _ = ax.axis('tight')



    # add thumbnails :) Displaying thumbnails is something I have commented out. 

#     from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#     for i in range(len(images)):

#         image = plt.imread(images[i])

#         im = OffsetImage(image, zoom=zoom)

#         bboxprops = dict(edgecolor=colors[i]) if colors is not None else None

#         ab = AnnotationBbox(im, x[i], xycoords='data',

#                             frameon=(bboxprops is not None),

#                             pad=0.02,

#                             bboxprops=bboxprops)

#         ax.add_artist(ab)

    return ax

%%time

import pickle



embeddings = pd.read_pickle('/kaggle/input/sample-face-crop/embeddings_face_clusters.pkl')

print(embeddings.shape)

embeddings.head()
# Convert the embeddings to columns

colnames = list()



for idx in range(512):

    colnames.append('colname_'+str(idx))

    

colnames;

embeddings[colnames] = pd.DataFrame(embeddings['embedding'].values.tolist(), index = embeddings.index)
#Convert to numpy array

embed_numpy = embeddings[colnames].to_numpy()

%%time

# PCA first to speed it up

x = PCA(n_components=50).fit_transform(embed_numpy)

%%time

tsne = TSNE(random_state = 99) # 

x = tsne.fit_transform(embed_numpy)
%%time

tsne50 = TSNE(random_state=99, n_components=50)

x50= tsne50.fit_transform(embed_numpy)
%%time 

dbscan = DBSCAN(eps=1.5, verbose=True ) #min_samples (default is 5)

clusters =  dbscan.fit_predict(x)

embeddings['RapidDBSCAN'] = clusters
embeddings.to_pickle('/kaggle/working/embeddings.pkl')
