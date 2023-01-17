



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import subplot

from skimage.color import rgb2hsv

from skimage import data

from skimage.color import rgb2gray

plt.rcParams["figure.figsize"] = (15,15)

import warnings

warnings.filterwarnings('ignore')

import numpy as np



def read_image(fn):

    # save the image

    image = plt.imread(fn)



    # display

    plt.subplot(2, 1, 1)

    plt.title('Original Image')

    plt.axis('off')

    plt.imshow(image)



    # Extract 2-D arrays of the RGB channels: red, blue, green

    red, blue, green = image[:, :, 0], image[:, :, 1], image[:, :, 2]



    # Flatten the 2-D arrays of the RGB channels into 1-D

    red_pixels = red.flatten()

    blue_pixels = blue.flatten()

    green_pixels = green.flatten()



    # Overlay histograms of the pixels of each color in the bottom subplot

    plt.subplot(2, 1, 2)

    plt.title('RBG Histogram of the image')

    plt.xlim((0, 256))

    sns.distplot(red_pixels, bins=64,  color='red', hist_kws=dict(edgecolor="k"))

    sns.distplot(blue_pixels, bins=64,  color='blue', hist_kws=dict(edgecolor="k"))

    sns.distplot(green_pixels, bins=64,  color='green', hist_kws=dict(edgecolor="k"))

    # Display the plot

    plt.show()
read_image('/kaggle/input/ferrari.jpg')
read_image('/kaggle/input/audi.jpg')
def color_to_bw(fn):

    image = plt.imread(fn)

    greyscale = rgb2gray(image)

    fig, axes = plt.subplots(1, 2, figsize=(30, 30))

    ax = axes.ravel()

    ax[0].imshow(image)

    ax[0].set_title("Original")

    ax[1].imshow(greyscale, cmap=plt.cm.gray)

    ax[1].set_title("Grayscale")



    fig.tight_layout()

    plt.show()

color_to_bw('/kaggle/input/ferrari.jpg')
def color_to_negative(fn):

    image = plt.imread(fn)

    negative =255- image # neg = (L-1) - img

    fig, axes = plt.subplots(1, 2, figsize=(30, 30))

    ax = axes.ravel()



    ax[0].imshow(image)

    ax[0].set_title("Original")

    ax[1].imshow(negative, cmap=plt.cm.gray)

    ax[1].set_title("negative")



    fig.tight_layout()

    plt.show()
color_to_negative('/kaggle/input/ferrari.jpg')
import numpy as np

import matplotlib.pyplot as plt



from skimage import color, data, restoration



astro = color.rgb2gray(plt.imread('/kaggle/input/ferrari.jpg'))

from scipy.signal import convolve2d as conv2

psf = np.ones((5, 5)) / 25

astro = conv2(astro, psf, 'same')

astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape)



deconvolved, _ = restoration.unsupervised_wiener(astro, psf)



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20),

                       sharex=True, sharey=True)



plt.gray()



ax[0].imshow(astro, vmin=deconvolved.min(), vmax=deconvolved.max())

ax[0].axis('off')

ax[0].set_title('Original in BW')



ax[1].imshow(deconvolved)

ax[1].axis('off')

ax[1].set_title('Enhanced')



fig.tight_layout()



plt.show()
from skimage import data, io, segmentation, color

from skimage.future import graph

import numpy as np





def _weight_mean_color(graph, src, dst, n):

    """Callback to handle merging nodes by recomputing mean color.



    The method expects that the mean color of `dst` is already computed.



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

        A dictionary with the `"weight"` attribute set as the absolute

        difference of the mean color between node `dst` and `n`.

    """



    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']

    diff = np.linalg.norm(diff)

    return {'weight': diff}





def merge_mean_color(graph, src, dst):

    """Callback called before merging two nodes of a mean color distance graph.



    This method computes the mean color of `dst`.



    Parameters

    ----------

    graph : RAG

        The graph under consideration.

    src, dst : int

        The vertices in `graph` to be merged.

    """

    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']

    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']

    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /

                                      graph.nodes[dst]['pixel count'])





img = plt.imread('/kaggle/input/ferrari.jpg')

labels = segmentation.slic(img, compactness=30, n_segments=400)

g = graph.rag_mean_color(img, labels)



labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,

                                   in_place_merge=True,

                                   merge_func=merge_mean_color,

                                   weight_func=_weight_mean_color)



out = color.label2rgb(labels2, img, kind='avg', bg_label=0)

out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))

io.imshow(out)

io.show()
from skimage import data, segmentation, filters, color

from skimage.future import graph

from matplotlib import pyplot as plt





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



img = plt.imread('/kaggle/input/ferrari.jpg')

edges = filters.sobel(color.rgb2gray(img))

labels = segmentation.slic(img, compactness=30, n_segments=400)

g = graph.rag_boundary(labels, edges)



graph.show_rag(labels, g, img)

plt.title('Initial RAG')



labels2 = graph.merge_hierarchical(labels, g, thresh=0.08, rag_copy=False,

                                   in_place_merge=True,

                                   merge_func=merge_boundary,

                                   weight_func=weight_boundary)



graph.show_rag(labels, g, img)

plt.title('RAG after hierarchical merging')



plt.figure()

out = color.label2rgb(labels2, img, kind='avg', bg_label=0)

plt.imshow(out)

plt.title('Final segmentation')



plt.show()