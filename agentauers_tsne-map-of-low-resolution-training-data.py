import math, re, os

import tensorflow as tf

import numpy as np

from kaggle_datasets import KaggleDatasets

import PIL

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
LABELED_TFREC_FORMAT = {

    "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

    "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

}



def parseItem(item):

    rec = tf.io.parse_single_example(item, LABELED_TFREC_FORMAT)

    img = rec['image']

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (10,10), antialias=True)

    img = tf.cast(img, tf.uint8)

    return img



filename_pattern = KaggleDatasets().get_gcs_path() + '/tfrecords-jpeg-192x192/train/*.tfrec'

filenames = tf.io.gfile.glob(filename_pattern)

ds = tf.data.TFRecordDataset(filenames).map(parseItem)

thumbnails = np.stack([img for img in iter(ds)])



thumbnails.shape
mosaic = PIL.Image.new(mode='RGB', size=(800, 200))

i = 0

for ix in range(80):

    for iy in range(20):

        t = PIL.Image.fromarray(thumbnails[i,:,:,:])

        mosaic.paste(t, (ix*10, iy*10))

        i = i + 1



mosaic
(n, dimx, dimy, chan) = thumbnails.shape

data = thumbnails.reshape((n, dimx * dimy * chan))

data = PCA(n_components=32).fit_transform(data)



embedding = TSNE(n_components=2, verbose=2, perplexity=250).fit_transform(data)
(n, w) = embedding.shape



img = PIL.Image.new(mode='RGB', size=(800, 800))    

for i in range(n):

    img2 = PIL.Image.fromarray(thumbnails[i])

    x = math.floor(embedding[i,0]*16 + 400)

    y = math.floor(embedding[i,1]*16 + 400)

    img.paste(img2, (x, y))



img