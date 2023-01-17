# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import pydicom

import glob

from PIL import Image

import matplotlib.pyplot as plt





inputdir = '../input/sample images/'

outdir = './'



fig, axs = plt.subplots(2, 5, figsize=(30, 10))

test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]

for f,ax in zip(test_list,axs.flatten()):  

    ds = pydicom.read_file( inputdir + f) # read dicom image

    img = ds.pixel_array # get image array

    img_mem = Image.fromarray(img)

    ax.imshow(img_mem)