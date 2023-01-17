# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from matplotlib.image import imread
image_folder = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']

nimgs = {}

for i in image_folder:

    nimages = len(os.listdir('/kaggle/input/multiclass-weather-dataset/dataset/'+i+'/'))

    nimgs[i]=nimages

plt.figure(figsize=(10, 8))

plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')

plt.xticks(range(len(nimgs)), list(nimgs.keys()))

plt.title('Distribution of different classes of Dataset')

plt.show()
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import ImageDataGenerator

from numpy import expand_dims
# load the image

img = load_img('/kaggle/input/multiclass-weather-dataset/dataset/cloudy/cloudy219.jpg')

plt.imshow(img)
data = img_to_array(img)

samples = expand_dims(data, 0)
datagen = ImageDataGenerator(

    horizontal_flip=True,

    vertical_flip=True

)
it = datagen.flow(samples, batch_size=1)

# generate samples and plot

for i in range(9):

    # define subplot

    plt.subplot(330 + 1 + i)

    # generate batch of images

    batch = it.next()

    # convert to unsigned integers for viewing

    image = batch[0].astype('uint8')

    # plot raw pixel data

    plt.imshow(image)

# show the figure

plt.show()