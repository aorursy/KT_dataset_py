# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

import codecs

import subprocess



import PIL

from PIL import Image

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True



# your image path

img_dir = '../input/images_sample/'



def white_count(img_url):

    im = Image.open(img_url)  

    w, h = im.size  

    colors = im.getcolors(w*h)

    return im.getcolors(w*h)[0][0] / (w * h * 1.0)



for listing_dir in os.listdir(img_dir):

    if len(listing_dir) != 7 or not listing_dir.isdigit():

        continue

    

    for listing_img in os.listdir(img_dir + listing_dir):

        if listing_img == '.DS_Store' or listing_img == listing_dir:

            continue

        white_scale = white_count(img_dir + listing_dir + '/' + listing_img)

        if white_scale > 0.6:

            print(listing_img, white_scale)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg



plt.figure(figsize = (12, 10))



img1 = mpimg.imread('../input/images_sample/6811974/6811974_197bb9515b3d7929c2848e61a050ad1a.jpg')

plt.subplot(1, 2, 1)

plt.imshow(img1) 



img2 = mpimg.imread('../input/images_sample/6811958/6811958_bb863a4184a1e085f0c55e0172767abd.jpg')

plt.subplot(1, 2, 2)

plt.imshow(img2)