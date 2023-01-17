import os

import gzip

import json



import PIL



import numpy as np

import pandas as pd

import skimage

pd.set_option('max_columns', 50)

pd.set_option('max_rows', 1000)



import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (8, 6)



from pandas.io.parsers import read_csv

from sklearn.utils import shuffle



from IPython.core.display import display, HTML, Image
def img_to_array(img):

    return np.array(img.getdata()).reshape(img.width, img.width, 3) / 255





def trim(im):

    """trim black margin, http://stackoverflow.com/questions/10615901/trim-whitespace-using-pil"""

    bg = PIL.Image.new(im.mode, im.size, im.getpixel((0,0)))

    diff = PIL.ImageChops.difference(im, bg)

    diff = PIL.ImageChops.add(diff, diff, 2.0, -20)

    bbox = diff.getbbox()

    if bbox:

        return im.crop(bbox)





def calc_thumbnail_size(img):

    """calculate thumbnail size with constant aspect ratio"""

    width, length = img.size

    ratio = width / length



    # for some reason, if it's exactly 224, then thumnailed image is 223

    dim = 224 + 1          # output dim

    if ratio > 1:

        size = (dim * ratio, dim)

    else:

        size = (dim, dim / ratio)

#     print(size)

    return size





def calc_crop_coords(img):

    """crop to square of desired dimension size"""

    dim = 224

    width, length = img.size

    left = 0

    right = width

    bottom = length

    top = 0

    if width > dim:

        delta = (width - dim) / 2

        left = delta

        right = width - delta

    if length > dim:

        delta = (length - dim) / 2

        top = delta

        bottom = length - delta

    return (left, top, right, bottom)





def preprocess(img):

    img = trim(img)

    tsize = calc_thumbnail_size(img)

    img.thumbnail(tsize)

    crop_coords = calc_crop_coords(img)

    img = img.crop(crop_coords)

    return img
df = pd.read_csv('../input/trainLabels.csv')
df.level.value_counts().to_frame(name='count').T
df = df.query('image in {0}'.format([_.replace('.jpeg', '') for _ in os.listdir('../input/')]))
data_dir = '../input/'
PIL.__version__
imgs_with_label = []

n_samples = 5

for i in range(5):

    _vals = df.query('level == {0}'.format(i)).sample(n_samples).image.apply(

        lambda v: (os.path.join(data_dir, v) + '.jpeg', i)).values.tolist()

    imgs_with_label.extend(_vals)



fig, axes = plt.subplots(5, 5, figsize=(16, 16))

axes = axes.ravel()

for k, (img, label) in enumerate(imgs_with_label):

    im = PIL.Image.open(img)

    im = preprocess(im)

    ax = axes[k]

    ax.imshow(img_to_array(im))

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.grid(False)

    if k % 5 == 0:

        ax.set_ylabel('level = {0}'.format(label))

    ax.set_title(os.path.basename(img))
imgs_with_label = []

n_samples = 5

for i in range(5):

    _vals = df.query('level == {0}'.format(i)).sample(n_samples).image.apply(

        lambda v: (os.path.join(data_dir, v) + '.jpeg', i)).values.tolist()

    imgs_with_label.extend(_vals)



fig, axes = plt.subplots(5, 5, figsize=(16, 16))

axes = axes.ravel()

for k, (img, label) in enumerate(imgs_with_label):

    im = PIL.Image.open(img)

    im = preprocess(im)

    ax = axes[k]

    ax.imshow(img_to_array(im))

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.grid(False)

    if k % 5 == 0:

        ax.set_ylabel('level = {0}'.format(label))

    ax.set_title(os.path.basename(img))
imgs_with_label = []

n_samples = 5

for i in range(5):

    _vals = df.query('level == {0}'.format(i)).sample(n_samples).image.apply(

        lambda v: (os.path.join(data_dir, v) + '.jpeg', i)).values.tolist()

    imgs_with_label.extend(_vals)



fig, axes = plt.subplots(5, 5, figsize=(16, 16))

axes = axes.ravel()

for k, (img, label) in enumerate(imgs_with_label):

    im = PIL.Image.open(img)

    im = preprocess(im)

    ax = axes[k]

    ax.imshow(img_to_array(im))

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.grid(False)

    if k % 5 == 0:

        ax.set_ylabel('level = {0}'.format(label))

    ax.set_title(os.path.basename(img))