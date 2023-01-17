# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import sys

import os

import subprocess



# Make sure you have all of these packages installed, e.g. via pip

import numpy as np

import pandas as pd

import seaborn as sns

#import rasterio

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import scipy

from skimage import io

from scipy import ndimage

from IPython.display import display

%matplotlib inline



!ls -ha ../input
PLANET_KAGGLE_ROOT = os.path.abspath("../input/")

PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, './train-jpg')

PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, './train.csv')

assert os.path.exists(PLANET_KAGGLE_ROOT)

assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)

assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)
labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)

labels_df.head()
# Build list with unique labels

label_list = []

for tag_str in labels_df.tags.values:

    labels = tag_str.split(' ')

    for label in labels:

        if label not in label_list:

            label_list.append(label)

           
# Add onehot features for every label

for label in label_list:

    labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

# Display head

labels_df.head()
# Histogram of label instances

labels_df[label_list].sum().sort_values().plot.bar()
def make_cooccurence_matrix(labels):

    numeric_df = labels_df[labels]; 

    c_matrix = numeric_df.T.dot(numeric_df)

    sns.heatmap(c_matrix)

    return c_matrix

    

# Compute the co-ocurrence matrix

make_cooccurence_matrix(label_list)
# Each image should have exactly one weather label:

weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']

make_cooccurence_matrix(weather_labels)
# But the land labels may overlap:

land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation']

make_cooccurence_matrix(land_labels)
# The rarer labels have very little overlap:

rare_labels = [l for l in label_list if labels_df[label_list].sum()[l] < 2000]

make_cooccurence_matrix(rare_labels)
# Let's display an image and visualize the pixel values. Here we will pick an image, 

# load every single single band, then create RGB stack. These raw images are

# 16-bit (from 0 to 65535), and contain red, green, blue, and Near infrared (NIR) channels.

#In this example, we are discarding the NIR band just to simplify the steps to 

# visualize the image. However, you should probably keep it for ML classification.





from six import string_types



def sample_images(tags, n=None):

    """Randomly sample n images with the specified tags."""

    condition = True

    if isinstance(tags, string_types):

        raise ValueError("Pass a list of tags, not a single tag.")

    for tag in tags:

        condition = condition & labels_df[tag] == 1

    if n is not None:

        return labels_df[condition].sample(n)

    else:

        return labels_df[condition]  
def find_image(filename):

    """Return a 4D (r, g, b, nir) numpy array with the data in the specified TIFF filename."""

    for dirname in os.listdir(PLANET_KAGGLE_ROOT):

        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))

        if os.path.exists(path):

            print('Found image {}'.format(path))

            return path

        else:

            print('Cannot find image at {}'.format(path))



def load_image(filename):

    path = find_image(filename)

    if path:

        return io.imread(path)

    else:

        print('Load failed: could not find image {}'.format(path))

        

def plot_rgbn_histo(r, g, b, n):

    for slice_, name, color in ((r,'r', 'red'),(g,'g', 'green'),(b,'b', 'blue'), (nir, 'nir', 'magenta')):

        plt.hist(slice_.ravel(), bins=100, 

                 range=[0,rgb_image.max()], 

                 label=name, color=color, histtype='step')

    plt.legend()



def sample_to_fname(sample_df, row_idx, suffix='tif'):

    '''Given a dataframe of sampled images, get the

    corresponding filename.'''

    fname = sample_df.get_value(sample_df.index[row_idx], 'image_name')

    return '{}.{}'.format(fname, suffix)        
# Let's look at an individual image. First, we'll plot a histogram of pixel values in each channel.

# Note how the intensities are distributed in a relatively narrow region of the dynamic range

s = sample_images(['primary', 'water', 'road'], n=1)

image_path = sample_to_fname(s, 0)

rgbn_image = load_image(image_path)

rgb_image = rgbn_image[:,:,:3]

r, g, b, nir = rgbn_image[:, :, 0], rgbn_image[:, :, 1], rgbn_image[:, :, 2], rgbn_image[:, :, 3]



# plot a histogram of rgbn values

plot_rgbn_histo(r, g, b, nir)
# We can look at each channel individually:

fig = plt.figure()

fig.set_size_inches(12, 4)

for i, (x, c) in enumerate(((r, 'r'), (g, 'g'), (b, 'b'), (nir, 'near-ir'))):

    a = fig.add_subplot(1, 4, i+1)

    a.set_title(c)

    plt.imshow(x)
plt.imshow(rgb_image)
# Calibrate colors for visual inspection

#  here we will employ the JPEG images provided in the data set,

# which have already been color-corrected.

# Get a list of reference images to extract data from:



# Pull a list of 20000 image names

jpg_list = os.listdir(PLANET_KAGGLE_JPEG_DIR)[:20000]

# Select a random sample of 100 among those

np.random.shuffle(jpg_list)

jpg_list = jpg_list[:100]



print(jpg_list)
# Read each image (8-bit RGBA) and dump the pixels values to ref_colors,

# which contains buckets for R, G and B

ref_colors = [[],[],[]]

for _file in jpg_list:

    # keep only the first 3 bands, RGB

    _img = mpimg.imread(os.path.join(PLANET_KAGGLE_JPEG_DIR, _file))[:,:,:3]

    # Flatten 2-D to 1-D

    _data = _img.reshape((-1,3))

    # Dump pixel values to aggregation buckets

    for i in range(3): 

        ref_colors[i] = ref_colors[i] + _data[:,i].tolist()

    

ref_colors = np.array(ref_colors)
# Visualize the histogram of the reference data

for i,color in enumerate(['r','g','b']):

    plt.hist(ref_colors[i], bins=30, range=[0,255], label=color, color=color, histtype='step')

plt.legend()

plt.title('Reference color histograms')
# Compute the mean and variance for each channel in the reference data

ref_means = [np.mean(ref_colors[i]) for i in range(3)]

ref_stds = [np.std(ref_colors[i]) for i in range(3)]
# And now, we have a function that can calibrate any raw image reasonably well:

def calibrate_image(rgb_image):

    # Transform test image to 32-bit floats to avoid 

    # surprises when doing arithmetic with it 

    calibrated_img = rgb_image.copy().astype('float32')



    # Loop over RGB

    for i in range(3):

        # Subtract mean 

        calibrated_img[:,:,i] = calibrated_img[:,:,i]-np.mean(calibrated_img[:,:,i])

        # Normalize variance

        calibrated_img[:,:,i] = calibrated_img[:,:,i]/np.std(calibrated_img[:,:,i])

        # Scale to reference 

        calibrated_img[:,:,i] = calibrated_img[:,:,i]*ref_stds[i] + ref_means[i]

        # Clip any values going out of the valid range

        calibrated_img[:,:,i] = np.clip(calibrated_img[:,:,i],0,255)



    # Convert to 8-bit unsigned int

    return calibrated_img.astype('uint8')
# Visualize the color histogram of the newly calibrated test image, and note that it's more

# evenly distributed throughout the dynamic range, and is closer to the reference data.

test_image_calibrated = calibrate_image(rgb_image)

for i,color in enumerate(['r','g','b']):

    plt.hist(test_image_calibrated[:,:,i].ravel(), bins=30, range=[0,255], 

             label=color, color=color, histtype='step')

plt.legend()

plt.title('Calibrated image color histograms')
plt.imshow(test_image_calibrated)
#Putting it all together, to show several images with your tags of choice:

sampled_images = sample_images(['clear', 'road', 'water'], n=1)

for i in range(len(sampled_images)):

    tif = sample_to_fname(sampled_images, i, 'tif')

    jpg = sample_to_fname(sampled_images, i, 'jpg')



    try:

        tif_img = load_image(tif)[:,:,:3]

        jpg_img = load_image(jpg)[:,:,:3]



        fig = plt.figure()

        plt.imshow(calibrate_image(tif_img))



        fig = plt.figure()

        plt.imshow(calibrate_image(jpg_img))

                

    except:

        continue
# Image clustering

# For our purpose we will just use the pixel intensities and compute pairwise distances



import cv2

from glob import glob

image_paths = sorted(glob('../input/train-jpg/*.jpg'))[0:1000]

image_names = list(map(lambda row: row.split("/")[-1][:-4], image_paths))



n_imgs = 600



all_imgs = []



for i in range(n_imgs):

    img = plt.imread(image_paths[i])

    img = cv2.resize(img, (100, 100), cv2.INTER_LINEAR).astype('float')

#    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float')

    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)

    img = img.reshape(1, -1)

    all_imgs.append(img)



img_mat = np.vstack(all_imgs)

img_mat.shape
# We can see frmo the line spectrum in the clustermap, that there are a few images 

# that are very dissimilar to all other images by using the pixel intensities.

# Also there is a block-like structure to it, maybe that already tells us

#something about the tags themselves.

from scipy.spatial.distance import pdist, squareform



sq_dists = squareform(pdist(img_mat))

print(sq_dists.shape)

sns.clustermap(

    sq_dists,

    figsize=(12,12),

    cmap=plt.get_cmap('viridis')

)
# Let's have a look at t-SNE embedding of the images to get a nice visualization

# of the distances in three dimensions.

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from sklearn.manifold import TSNE

tsne = TSNE(

    n_components=3,

    init='random', # pca

    random_state=101,

    method='barnes_hut',

    n_iter=500,

    verbose=2

).fit_transform(img_mat)
trace1 = go.Scatter3d(

    x=tsne[:,0],

    y=tsne[:,1],

    z=tsne[:,2],

    mode='markers',

    marker=dict(

        sizemode='diameter',

        #color = preprocessing.LabelEncoder().fit_transform(all_image_types),

        #colorscale = 'Portland',

        #colorbar = dict(title = 'images'),

        line=dict(color='rgb(255, 255, 255)'),

        opacity=0.9

    )

)



data=[trace1]

layout=dict(height=800, width=800, title='3D embedding of images')

fig=dict(data=data, layout=layout)

py.iplot(fig, filename='3DBubble')
mask = np.zeros_like(sq_dists, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# upper triangle of matrix set to np.nan

sq_dists[np.triu_indices_from(mask)] = np.nan

sq_dists[0, 0] = np.nan



fig = plt.figure(figsize=(12,8))

# maximally dissimilar image

ax = fig.add_subplot(1,2,1)

maximally_dissimilar_image_idx = np.nanargmax(np.nanmean(sq_dists, axis=1))

plt.imshow(plt.imread(image_paths[maximally_dissimilar_image_idx]))

plt.title('maximally dissimilar')



# maximally similar image

ax = fig.add_subplot(1,2,2)

maximally_similar_image_idx = np.nanargmin(np.nanmean(sq_dists, axis=1))

plt.imshow(plt.imread(image_paths[maximally_similar_image_idx]))

plt.title('maximally similar')



# # now compute the mean image

#ax = fig.add_subplot(1,3,3)

#mean_img = gray_imgs_mat.mean(axis=0).reshape(rescaled_dim, rescaled_dim, 3)

#plt.imshow(cv2.normalize(mean_img, None, 0.0, 1.0, cv2.NORM_MINMAX))

#plt.title('mean image')
from sklearn.manifold import TSNE

tsne = TSNE(

    n_components=2,

    init='random', # pca

    random_state=101,

    method='barnes_hut',

    n_iter=500,

    verbose=2

).fit_transform(img_mat)
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x, y, images, ax=None, zoom=0.1):

    ax = plt.gca()

    images = [OffsetImage(image, zoom=zoom) for image in images]

    artists = []

    for x0, y0, im0 in zip(x, y, images):

        ab = AnnotationBbox(im0, (x0, y0), xycoords='data', frameon=False)

        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))

    ax.autoscale()

    #return artists



nimgs = 500

plt.figure(figsize=(13,10))

imscatter(tsne[0:nimgs,0], tsne[0:nimgs,1], [plt.imread(image_paths[i]) for i in range(nimgs)])
# We're now going to compute the NDVI for a few images and rank them by it to see

#how well we can identify the amount of vegetation in the image.

image_paths = sorted(glob('../input/train-tif/*.tif'))[0:1000]

imgs = [io.imread(path) / io.imread(path).max() for path in image_paths]

#r, g, b, nir = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]

ndvis = [(img[:,:,3] - img[:,:,0])/((img[:,:,3] + img[:,:,0])) for img in imgs]
plt.figure(figsize=(12,8))

plt.subplot(121)

plt.imshow(ndvis[32], cmap='jet')

plt.colorbar()

plt.title('NDVI index of cloudy image')

plt.subplot(122)

plt.imshow(imgs[32])
plt.figure(figsize=(12,8))

plt.subplot(121)

plt.imshow(ndvis[800], cmap='jet')

plt.colorbar()

plt.title('NDVI index of image with lots of vegetation')

plt.subplot(122)

plt.imshow(imgs[800])
mndvis = np.nan_to_num([ndvi.mean() for ndvi in ndvis])

plt.figure(figsize=(12,8))

sns.distplot(mndvis)

plt.title('distribution of mean NDVIs')
sorted_idcs = np.argsort(mndvis)

print(len(sorted_idcs))

plt.figure(figsize=(12,8))

plt.subplot(221)

plt.imshow(ndvis[sorted_idcs[0]], cmap='jet')

plt.subplot(222)

plt.imshow(ndvis[sorted_idcs[50]], cmap='jet')

plt.subplot(223)

plt.imshow(ndvis[sorted_idcs[-30]], cmap='jet')

plt.subplot(224)

plt.imshow(ndvis[sorted_idcs[-11]], cmap='jet')
import tensorflow as tf

from tensorflow.python.framework import ops

ops.reset_default_graph()



# Introduce tensors in tf



# Get graph handle

sess = tf.Session()

my_tensor = tf.zeros([1,20])



# Declare a variable

my_var = tf.Variable(tf.zeros([1,20]))



# Different kinds of variables

row_dim = 2

col_dim = 3 



# Zero initialized variable

zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))



# One initialized variable

ones_var = tf.Variable(tf.ones([row_dim, col_dim]))



# shaped like other variable

sess.run(zero_var.initializer)

sess.run(ones_var.initializer)

zero_similar = tf.Variable(tf.zeros_like(zero_var))

ones_similar = tf.Variable(tf.ones_like(ones_var))



sess.run(ones_similar.initializer)

sess.run(zero_similar.initializer)



# Fill shape with a constant

fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))



# Create a variable from a constant

const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))

# This can also be used to fill an array:

const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))



# Sequence generation

linear_var = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) # Generates [0.0, 0.5, 1.0] includes the end



sequence_var = tf.Variable(tf.range(start=6, limit=15, delta=3)) # Generates [6, 9, 12] doesn't include the end



# Random Numbers



# Random Normal

#rnorm_var = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)



# Initialize operation

initialize_op = tf.global_variables_initializer()



# Add summaries to tensorboard

#merged = tf.merge_all_summaries()



# Initialize graph writer:

#writer = tf.train.SummaryWriter("/tmp/variable_logs", sess.graph_def)



# Run initialization of variable

sess.run(initialize_op)