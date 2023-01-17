import matplotlib.pyplot as plt

import numpy as np

import cv2

import pandas as pd

from shapely.wkt import loads as wkt_loads

import tifffile as tiff

import os

import random

from sklearn.metrics import jaccard_similarity_score

from shapely.geometry import MultiPolygon, Polygon

import shapely.wkt

import shapely.affinity

from collections import defaultdict



image_id='6030_2_2'

os.system("pwd")

print("-----------------------------------------")

inDir = "../input/"

filename_M = os.path.join(inDir, 'sixteen_band', '{}_M.tif'.format(image_id))

img_M = tiff.imread(filename_M)

img_M2 = np.rollaxis(img_M, 0, 3)

img_M2_resize = cv2.resize(img_M2, (512, 512)) 



filename_A = os.path.join(inDir, 'sixteen_band', '{}_A.tif'.format(image_id))

img_A = tiff.imread(filename_A)

img_A2 = np.rollaxis(img_A, 0, 3)

img_A2_resize = cv2.resize(img_A2, (512, 512)) 



filename_P = os.path.join(inDir, 'sixteen_band', '{}_P.tif'.format(image_id))

img_P2 = tiff.imread(filename_P)

img_P2_resize = cv2.resize(img_P2, (512, 512)) 





filename_RGB = os.path.join(inDir, 'three_band', '{}.tif'.format(image_id))

img_RGB = tiff.imread(filename_RGB)

img_RGB2 = np.rollaxis(img_RGB, 0, 3)

img_RGB2_resize = cv2.resize(img_RGB2, (512, 512)) 



Array = np.zeros((img_RGB2_resize.shape[0],img_RGB2_resize.shape[1],20), 'uint8')

Array[..., 0:3] = img_RGB2_resize

Array[..., 3] = img_P2_resize

Array[..., 4:12] = img_M2_resize

Array[..., 12:21] = img_A2_resize

Array.shape #(512, 512, 20)