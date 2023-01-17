import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



from sklearn.decomposition import PCA
os.listdir('/kaggle/input/sample-face-crop')
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



    # add thumbnails :)

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
import pickle



embeddings = pd.read_pickle('/kaggle/input/sample-face-crop/embeddings_face_clusters.pkl')

print(embeddings.shape)

embeddings.head()
# _ = scatter_thumbnails(embeddings.embedding.tolist(), embeddings.faceFile.tolist())

# plt.title('Facial Embeddings - Principal Component Analysis')

# plt.show()
%%time

from sklearn.manifold import TSNE
%%time

# PCA first to speed it up

x = PCA(n_components=50).fit_transform(embeddings['embedding'].tolist())

%%time

x = TSNE(perplexity=50,

         n_components=3).fit_transform(x)
# _ = scatter_thumbnails(x, embeddings.faceFile.tolist(), zoom=0.06)

# plt.title('3D t-Distributed Stochastic Neighbor Embedding')

# plt.show()
!pip install -q hdbscan

import hdbscan
import sklearn.cluster as cluster


def plot_clusters(data, algorithm, *args, **kwds):

    labels = algorithm(*args, **kwds).fit_predict(data)

    #palette = sns.color_palette('deep', np.max(labels) + 1)

    #colors = [palette[x] if x >= 0 else (0,0,0) for x in labels]

    #ax = scatter_thumbnails(x, df.face.tolist(), 0.06, colors)

    #plt.title(f'Clusters found by {algorithm.__name__}')

    return labels



# clusters = plot_clusters(x, hdbscan.HDBSCAN, alpha=1.0, min_cluster_size=2, min_samples=1)

clusters = plot_clusters(x, cluster.DBSCAN, n_jobs=-1, eps=1.0, min_samples=1)

embeddings['cluster'] = clusters
print(type(x))

print(x.shape)
embeddings['TSNE'] = x
embeddings.head()
embeddings.to_pickle('/kaggle/working/embedding_clusters.pkl')