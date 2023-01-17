import os

import numpy as np

import tifffile as tiff

import cv2

import matplotlib.pyplot as plt

from skimage.segmentation import slic, mark_boundaries



def stretch_8bit(bands, lower_percent=2, higher_percent=98):

    out = np.zeros_like(bands)

    for i in range(3):

        a = 0 #np.min(band)

        b = 255  #np.max(band)

        c = np.percentile(bands[:,:,i], lower_percent)

        d = np.percentile(bands[:,:,i], higher_percent)        

        t = a + (bands[:,:,i] - c) * (b - a) / (d - c)    

        t[t<a] = a

        t[t>b] = b

        out[:,:,i] =t

    return out.astype(np.uint8)    

    

def RGB(image_id):

    filename = os.path.join('..', 'input', 'three_band', '{}.tif'.format(image_id))

    img = tiff.imread(filename)

    img = np.rollaxis(img, 0, 3)    

    return img

    

def M(image_id):

    filename = os.path.join('..', 'input', 'sixteen_band', '{}_M.tif'.format(image_id))

    img = tiff.imread(filename)    

    img = np.rollaxis(img, 0, 3)

    return img
    image_id = '6120_2_2'

    rgb = RGB(image_id)

    rgb1 = stretch_8bit(rgb)

    

    y1,y2,x1,x2 = 1000, 1600, 2000, 2600

    region = rgb1[y1:y2, x1:x2, :]

    plt.figure()

    plt.imshow(region)

    m = M(image_id)    

    m = cv2.resize(m, tuple(reversed(rgb.shape[:2])))

    

    img = np.zeros_like(rgb)

    img[:,:,0] = m[:,:,6] #nir1

    img[:,:,1] = m[:,:,4] #red

    img[:,:,2] = rgb[:,:,2] #blue

    img = stretch_8bit(img)

    region = img[y1:y2, x1:x2, :]

    plt.figure()

    plt.imshow(region)

    blue_mask = cv2.inRange(region, np.array([15,115,200]), np.array([80,200,255]))    

    mask = cv2.bitwise_and(region, region, mask=blue_mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)    

    plt.figure()

    plt.imshow(mask, cmap='gray')

    segments = slic(region, n_segments=100, compactness=20.0, 

                    max_iter=10, sigma=5, spacing=None, multichannel=True, 

                    convert2lab=True, enforce_connectivity=False, 

                    min_size_factor=10, max_size_factor=3, slic_zero=False)

    boundaries = mark_boundaries(region, segments, color=(0,255,0))

    plt.figure()

    plt.imshow(boundaries)

    out = np.zeros_like(mask)

    for i in range(np.max(segments)):

        s = segments == i

        s_size = np.sum(s)

        s_count = np.sum([1 for x in mask[s].ravel() if x>0])

        #print(s_count, s_size)

        if s_count > 0.1*s_size:

            out[s] = 255

        

    plt.figure()

    plt.imshow(out, cmap='gray')

    
    out2 = cv2.bitwise_and(region, region, mask=out)

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(region)

    ax[1].imshow(out2)