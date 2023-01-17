# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import tifffile

import cv2



pp = tifffile.imread('../input/three_band/6120_2_2.tif').transpose((1,2,0)).astype(np.float32)

gray = tifffile.imread('../input/sixteen_band/6120_2_2_P.tif').astype(np.float32)

mm = tifffile.imread('../input/sixteen_band/6120_2_2_M.tif').transpose((1,2,0)).astype(np.float32)



pp.shape, mm.shape



mm2 = cv2.resize(mm,(pp.shape[1],pp.shape[0]),interpolation=cv2.INTER_CUBIC)

warp_mode = cv2.MOTION_EUCLIDEAN

warp_matrix = np.eye(2, 3, dtype=np.float32)

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-7)



#tifffile.imshow(mm[:,:,[4,2,1]])

img_orig = np.stack([mm2[:-2, :-4, 4], pp[2:, 4:, 1],pp[2:, 4:, 2]], axis=-1)



def stretch2(band, lower_percent=2, higher_percent=98):

    a = 0 #np.min(band)

    b = 255  #np.max(band)

    c = np.percentile(band, lower_percent)

    d = np.percentile(band, higher_percent)        

    out = a + (band - c) * (b - a) / (d - c)    

    out[out<a] = a

    out[out>b] = b

    return out



def adjust_contrast(x):    

    for i in range(3):

        x[:,:,i] = stretch2(x[:,:,i])

    return x.astype(np.uint8)

#tifffile.imshow(adjust_contrast(img_orig)[2000:2500,3000:])

#tifffile.imshow(adjust_contrast(pp)[2000:2500,3000:])



cv2.findTransformECC(pp[300:1900,300:2200,2], mm2[300:1900,300:2200,1], warp_matrix, warp_mode, criteria)
