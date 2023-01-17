# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import *

import matplotlib

import matplotlib.pyplot as plt

import cv2 

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def subtile_images(src_imgs_path, src_img_file_ext, dst_folder, dst_img_file_ext, dst_imgX, dst_imgY):

    ''' Method to split image files in smalled tile images. 

            src_imgs_path: flder path where all the image files to be splitted are present.

            src_img_file_ext: extension of the image files to be splitted

            dst_folder: folder path for saving splitted smaller images

            dst_img_file_ext: extension for splitted smaller images

            dst_imgX: height of tile

            dst_imgY: width of tile

    '''

    M = dst_imgX #500

    N = dst_imgY #500

    for img_file in Path(src_imgs_path).ls():

        if img_file.suffix != src_img_file_ext:

            continue

        im = plt.imread(img_file)

        tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]

        for i,t in enumerate(tiles):

            plt.imsave(str(Path(dst_folder)/f'{img_file.stem}_{i}{dst_img_file_ext}'), t)
# # Example call:

# # Split all the training images

# subtile_images('/home/../images', '.tif',

#                '/home/../trn_images', '.png',

#                500, 500)