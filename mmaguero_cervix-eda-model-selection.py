import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from skimage.io import imread, imshow

import cv2



%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from subprocess import check_output

print(check_output(["ls", "../input/train"]).decode("utf8"))
from glob import glob

basepath = '../input/train/'



all_cervix_images = []



for path in sorted(glob(basepath + "*")):

    cervix_type = path.split("/")[-1]

    cervix_images = sorted(glob(basepath + cervix_type + "/*"))

    all_cervix_images = all_cervix_images + cervix_images



all_cervix_images = pd.DataFrame({'imagepath': all_cervix_images})

all_cervix_images['filetype'] = all_cervix_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)

all_cervix_images['type'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)

all_cervix_images.head()
print('We have a total of {} images in the whole dataset'.format(all_cervix_images.shape[0]))

type_aggregation = all_cervix_images.groupby(['type', 'filetype']).agg('count')

type_aggregation_p = type_aggregation.apply(lambda row: 1.0*row['imagepath']/all_cervix_images.shape[0], axis=1)



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))



type_aggregation.plot.barh(ax=axes[0])

axes[0].set_xlabel("image count")

type_aggregation_p.plot.barh(ax=axes[1])

axes[1].set_xlabel("training size fraction")
fig = plt.figure(figsize=(12,8))



i = 1

for t in all_cervix_images['type'].unique():

    ax = fig.add_subplot(1,3,i)

    i+=1

    f = all_cervix_images[all_cervix_images['type'] == t]['imagepath'].values[0]

    plt.imshow(plt.imread(f))

    plt.title('sample for cervix {}'.format(t))
print(check_output(["ls", "../input/additional"]).decode("utf8"))
basepath = '../input/additional/'



all_cervix_images_a = []



for path in sorted(glob(basepath + "*")):

    cervix_type = path.split("/")[-1]

    cervix_images = sorted(glob(basepath + cervix_type + "/*"))

    all_cervix_images_a = all_cervix_images_a + cervix_images



all_cervix_images_a = pd.DataFrame({'imagepath': all_cervix_images_a})

all_cervix_images_a['filetype'] = all_cervix_images_a.apply(lambda row: row.imagepath.split(".")[-1], axis=1)

all_cervix_images_a['type'] = all_cervix_images_a.apply(lambda row: row.imagepath.split("/")[-2], axis=1)

all_cervix_images_a.head()
print('We have a total of {} images in the whole dataset'.format(all_cervix_images_a.shape[0]))

type_aggregation = all_cervix_images_a.groupby(['type', 'filetype']).agg('count')

type_aggregation_p = type_aggregation.apply(lambda row: 1.0*row['imagepath']/all_cervix_images_a.shape[0], axis=1)



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))



type_aggregation.plot.barh(ax=axes[0])

axes[0].set_xlabel("image count")

type_aggregation_p.plot.barh(ax=axes[1])

axes[1].set_xlabel("training size fraction")
fig = plt.figure(figsize=(12,8))



i = 1

for t in all_cervix_images_a['type'].unique():

    ax = fig.add_subplot(1,3,i)

    i+=1

    f = all_cervix_images_a[all_cervix_images_a['type'] == t]['imagepath'].values[0]

    plt.imshow(plt.imread(f))

    plt.title('sample for cervix {}'.format(t))
all_cervix_images_ = pd.concat( [all_cervix_images, all_cervix_images_a], join='outer' )

print(all_cervix_images_)
print('We have a total of {} images in the whole dataset'.format(all_cervix_images_.shape[0]))

type_aggregation = all_cervix_images_.groupby(['type', 'filetype']).agg('count')

type_aggregation_p = type_aggregation.apply(lambda row: 1.0*row['imagepath']/all_cervix_images_a.shape[0], axis=1)



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))



type_aggregation.plot.barh(ax=axes[0])

axes[0].set_xlabel("image count")

type_aggregation_p.plot.barh(ax=axes[1])

axes[1].set_xlabel("training size fraction")
fig = plt.figure(figsize=(12,8))



i = 1

for t in all_cervix_images_['type'].unique():

    ax = fig.add_subplot(1,3,i)

    i+=1

    f = all_cervix_images_[all_cervix_images_['type'] == t]['imagepath'].values[0]

    plt.imshow(plt.imread(f))

    plt.title('sample for cervix {}'.format(t))