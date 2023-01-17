# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from keras.preprocessing.image import img_to_array



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
image = mpimg.imread("../input/diceimages/dice-greyscale/dice-greyscale/775998487a5a1d351a4e4907d807c6e4.jpg")

plt.imshow(image)

plt.show()

image = mpimg.imread("../input/diceimages/dice-greyscale/dice-greyscale/775998487a5a1d351a4e4907d807c6e4.jpg")

plt.imshow(image)

plt.show()

print(image.shape)
from scipy import misc
def load_data(original_img_path,original_mask_path):

    original_img = cv2.imread(original_img_path)[:, :, ::-1]

    

    array_img = img_to_array(original_img)/255

    #image=array_img.reshape(330,500,3)

    original_img=misc.imresize(original_img, (330,500,3))

    original_mask = cv2.imread(original_mask_path)

    print(original_img.shape)
Dataset=os.listdir("../input/diceimages/dice-color/dice-color/")

#os.listdir("../input/diceimages/dice-color/dice-color/df071e12821a94515055b3182e31e471.jpg")

count=0

datafile=[]

for i in  Dataset:

    #print(i)

    #print(i.split('.',1)[0])

    if count>10:

        break;

    nstr="../input/diceimages/dice-color/dice-color/"+i.split('.',1)[0]+".jpg"

    #print(nstr)

    exists = os.path.isfile(nstr)

    if exists:

        #print(nstr)

        pstr="../input/diceimages/dice-color/dice-color/"+i

        

        image = mpimg.imread(pstr)

        

        plt.imshow(image)

        plt.show()

        load_data(nstr,pstr)

        image = mpimg.imread(nstr)

        plt.imshow(image)

        plt.show()

        datafile.append(i)

        count=count+1

    else:

        print("not found {}".format(nstr))