# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import io

import glob

import PIL

from PIL import Image

import pytesseract

import argparse

import cv2

import os

import matplotlib.pyplot as plt

from matplotlib import *

from pylab import *

import time

import ImageFilter
src_path = []



for path in glob.glob("/kaggle/input/ml_test/*.jpg"):

    src_path.append(path)
# data = image.open("https://storage.live.com/downloadfiles/V1/Zip?authKey=%21AIkuuQk7CMSNKRw&application=1141147648")
address = []

items_names = []

objects = []





def preprocessing():



    for items in src_path:



    # object-Recognition



        im = Image.open(items)

#         im = im.convert(mode='L')

#         thresh = 200

#         fn = lambda x : 255 if x > thresh else 0

#         im = im.convert('L').point(fn, mode='1')

        width, height = im.size

        print("Original_Resolution is : ", height, width)



    # Resize image



        new_width  = 4032

        new_height = 4032

        img = im.resize((new_width, new_height), Image.ANTIALIAS)

        width, height = img.size

        print("Resized_Resolution is : ", height, width)





        im = np.asarray(img)

        objects.append(im)



    # cropping Address only



        im = img.rotate(-90)

        im = im.crop((1000, 0, 4032, 270))

        im = np.asarray(im)

        address.append(im)



    # cropping items_names



        left = 0

        top = 2000

        right = 4032

        bottom = 4032



        im = img.rotate(-90)

        im = im.crop((left, top, right, bottom))

        im = np.asarray(im)

        items_names.append(im)



temp =  preprocessing()
for i in range(len(address)):

    imshow(address[i]);

    show();

    text = pytesseract.image_to_string(address[i])

    print(text)

    time.sleep(0.01)
for i in range(len(items_names)):

    imshow(items_names[i]);

    show()

    text = pytesseract.image_to_string(items_names[i])

    print(text)

    time.sleep(0.01)
print(len(address), len(items_names), len(objects), len(src_path))
# for i in range(len(address)):

#   plt.imshow(address[i]);



Image.open(address[8])
pytesseract.image_to_string(address[0])
print('--- Start recognize text from image ---')

print(get_string() )