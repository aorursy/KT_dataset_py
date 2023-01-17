import os

import numpy as np

import pandas as pd

import tifffile as tiff

import cv2

import matplotlib.pyplot as plt

from shapely.wkt import loads as wkt_loads



N_Cls = 10

inDir = '../input'

DF = pd.read_csv(inDir + '/train_wkt_v4.csv')

GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))

ISZ = 160

smooth = 1e-12



def stretch_n(bands, lower_percent=2, higher_percent=98):

    out = np.zeros_like(bands).astype(np.float32)

    for i in range(3):

        a = 0 

        b = 255 

        c = np.percentile(bands[:,:,i], lower_percent)

        d = np.percentile(bands[:,:,i], higher_percent)        

        t = a + (bands[:,:,i] - c) * (b - a) / (d - c)    

        t[t<a] = a

        t[t>b] = b

        out[:,:,i] =t

    return out.astype(np.uint8)    

    

def M(image_id):

    filename = os.path.join(inDir,'sixteen_band', '{}_M.tif'.format(image_id))

    img = tiff.imread(filename)    

    img = np.rollaxis(img, 0, 3)

    return img



image_id = '6120_2_3'

m = M(image_id)

img = np.zeros((837,851,3))

img[:,:,0] = m[:,:,4] #red

img[:,:,1] = m[:,:,2] #green

img[:,:,2] = m[:,:,1] #blue

plt.imshow(stretch_n(img))