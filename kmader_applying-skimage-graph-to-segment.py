import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils.io_utils import HDF5Matrix # for reading in data

import matplotlib.pyplot as plt # showing and rendering figures

# not needed in Kaggle, but required in Jupyter

%matplotlib inline 
td_path = '../input/interactive-segmentation-overview/training_data.h5'



img_data = HDF5Matrix(td_path, 'images')

stroke_data = HDF5Matrix(td_path, 'strokes')

seg_data = HDF5Matrix(td_path, 'segmentation')[:]



print('image data', img_data.shape)

print('stroke data', stroke_data.shape)

print('segmentation data', seg_data.shape)

_, _, _, img_chan = img_data.shape

_, _, _, stroke_chan = stroke_data.shape



# raw image data

path_df = pd.read_csv('../input/interactive-segmentation-overview/corrected_image_paths.csv')

path_df.drop(path_df.columns[0],axis=1,inplace=True)

path_df.sample(3)
from skimage import data, segmentation

from skimage.future import graph

import networkx as nx

from skimage.measure import regionprops



rand_idx = np.random.choice(range(img_data.shape[0]))

img = img_data[rand_idx]

labels = segmentation.slic(img, n_segments = 50, compactness=20.0)

rag = graph.rag_mean_color(img, labels, mode='distance')



fig, (ax_img, ax_slic, ax_seg) = plt.subplots(1, 3, figsize = (36, 12))

ax_img.imshow(img)

ax_slic.imshow(labels, cmap = plt.cm.BrBG)

ax_seg.imshow(seg_data[rand_idx][:,:,0])





pos_info = {c.label-1: np.array([c.centroid[1],c.centroid[0]]) for c in regionprops(labels+1)}

nx.draw(rag, pos = pos_info, ax = ax_slic)  # networkx draw()
# use the premade function

graph.show_rag(labels, rag, img, border_color = None)
rag.get_edge_data(0,1)
new_rag = rag.copy()

labels2 = graph.cut_threshold(labels, new_rag, thresh=0.10)

print(np.max(labels2-labels))

graph.show_rag(labels, new_rag, img, border_color = None)
def weight_boundary(graph, src, dst, n):

    """

    Handle merging of nodes of a region boundary region adjacency graph.



    This function computes the `"weight"` and the count `"count"`

    attributes of the edge between `n` and the node formed after

    merging `src` and `dst`.





    Parameters

    ----------

    graph : RAG

        The graph under consideration.

    src, dst : int

        The vertices in `graph` to be merged.

    n : int

        A neighbor of `src` or `dst` or both.



    Returns

    -------

    data : dict

        A dictionary with the "weight" and "count" attributes to be

        assigned for the merged node.



    """

    default = {'weight': 0.0, 'count': 0}



    count_src = graph[src].get(n, default)['count']

    count_dst = graph[dst].get(n, default)['count']



    weight_src = graph[src].get(n, default)['weight']

    weight_dst = graph[dst].get(n, default)['weight']



    count = count_src + count_dst

    return {

        'count': count,

        'weight': (count_src * weight_src + count_dst * weight_dst)/count

    }





def merge_boundary(graph, src, dst):

    """Call back called before merging 2 nodes.



    In this case we don't need to do any computation here.

    """

    pass

new_rag = rag.copy()

labels2 = graph.merge_hierarchical(labels, new_rag, thresh=0.0091, rag_copy=False,

                                   in_place_merge=True,

                                   merge_func=merge_boundary,

                                   weight_func=weight_boundary)



graph.show_rag(labels, new_rag, img, border_color='black', img_cmap = 'gist_heat', edge_cmap='nipy_spectral')
np.max(labels2-labels)