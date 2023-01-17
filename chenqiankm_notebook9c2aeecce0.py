__author__ = "n01z3"



import matplotlib.pyplot as plt

import numpy as np

import cv2

import pandas as pd

from shapely.wkt import loads as wkt_loads

import tifffile as tiff

import os

import random

from keras.models import Model

from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras import backend as K

from sklearn.metrics import jaccard_similarity_score

from shapely.geometry import MultiPolygon, Polygon

import shapely.wkt

import shapely.affinity

from collections import defaultdict



N_Cls = 10

DF = pd.read_csv('../input/train_wkt_v4.csv')

GS = pd.read_csv('../input/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

SB = pd.read_csv('../input/sample_submission.csv')

ISZ = 160

smooth = 1e-12
print("let's stick all imgs together")

s = 835

x = np.zeros((5 * s, 5 * s, 8))

y = np.zeros((5 * s, 5 * s, N_Cls))

ids = sorted(DF.ImageId.unique())

print(len(ids))

for i in range(5):

    for j in range(5):

        id = ids[5 * i + j]

        img = tiff.imread("../input/three_band/{}_M.tif".format(id))

        img = np.rollaxis(img, 0, 3)