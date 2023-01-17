# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input/stage_2_images/"))



# Any results you write to the current directory are saved as output.
# Convert DICOM to PNG via openCV

import cv2

import os

import pydicom

import glob



inputdir = '../input/stage_2_images'

outdir = './'

#os.mkdir(outdir)



test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]



for f in test_list:   

    ds = pydicom.read_file(inputdir + f) # read dicom image

    img = ds.pixel_array # get image array

    cv2.imwrite(outdir + f.replace('.dcm','.png'),img) # write png image
# Convert to PNG via PIL 

# https://github.com/python-pillow/Pillow

import os

import pydicom

import glob

from PIL import Image



inputdir = '../input/stage_2_images'

outdir = './'



test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]

#glob.glob(inputdir + './*.dcm')

for f in test_list:   

    ds = pydicom.read_file( inputdir + f) # read dicom image

    img = ds.pixel_array # get image array

    img_mem = Image.fromarray(img) # Creates an image memory from an object exporting the array interface

    img_mem.save(outdir + f.replace('.dcm','.png'))