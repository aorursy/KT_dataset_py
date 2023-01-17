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
#Reference link for the imagedatagenerator method

#https://theailearner.com/2019/07/06/imagedatagenerator-apply_transform-method/
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img



#Loading the image

img_org = load_img('/kaggle/input/messi.jpeg')



#Converting an image into array

img = img_to_array(img_org)



#Creating an instance of generator

datagen = ImageDataGenerator()

rotate = datagen.apply_transform(x=img, transform_parameters={'theta':40})

horiz_shift = datagen.apply_transform(x=img, transform_parameters={'tx':20})

vert_shift = datagen.apply_transform(x=img, transform_parameters={'ty':20})

shear = datagen.apply_transform(x=img, transform_parameters={'shear':30})

zoom_x = datagen.apply_transform(x=img, transform_parameters={'zx':0.2})

zoom_y = datagen.apply_transform(x=img, transform_parameters={'zy':0.3})

horiz_flip = datagen.apply_transform(x=img, transform_parameters={'flip_horizontal':True})

vert_flip = datagen.apply_transform(x=img, transform_parameters={'flip_vertical':True})





methods = [rotate,horiz_shift,vert_shift,shear,zoom_x,zoom_y,horiz_flip,vert_flip]

method = ['rotate','horiz_shift','vert_shift','shear','zoom_x','zoom_y','horiz_flip','vert_flip']
import matplotlib.pyplot as plt



plt.figure(0, figsize=(16,10))

for i in range(1,9):

    plt.subplot(2,4,i)

    plt.imshow(array_to_img(methods[i-1]))

    plt.title(method[i-1])

    