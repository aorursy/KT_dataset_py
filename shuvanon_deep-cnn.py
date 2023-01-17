# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf #Deep learning library



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

images = data.iloc[:,1:].values

images = images.astype(np.float)



images = np.multiply(images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(images.shape))



imageSize=images.shape[1]

print("Image Size = {0}".format(imageSize))



image_width = image_height = np.ceil(np.sqrt(imageSize)).astype(np.uint8)
IMAGE_TO_DISPLAY = 40 

def display(image):

    this_image=image.reshape(image_width,image_height)

    

    plt.axis('off')

    plt.imshow(this_image,cmap=cm.binary)



display(images[IMAGE_TO_DISPLAY])





label = data[[0]].values.ravel()

print ('label[{0}] => {1}'.format(IMAGE_TO_DISPLAY,label[IMAGE_TO_DISPLAY]))