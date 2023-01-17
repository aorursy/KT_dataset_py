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

        #print(os.path.join(dirname, filename))

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

pil_im = Image.open('../input/logocanal/LOGO PNG.png')

pil_im
mypath='../input/starwars-images/star-wars-images/star_wars/filtered/train/BB-8/'



from os import listdir

from os.path import isfile, join

bb8 = [f for f in listdir(mypath) if isfile(join(mypath, f))]

class_bb8 = ['bb8' for f in listdir(mypath) if isfile(join(mypath, f))]
import pandas as pd



df_star_wars=pd.DataFrame(bb8, columns=['filename'])

df_star_wars['class']=class_bb8
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_generator=ImageDataGenerator()

it=data_generator.flow_from_dataframe(df_star_wars, directory=mypath)
images,labels=it.next()
import matplotlib.pyplot as plt

import matplotlib.image as mpimg



plt.imshow(images[1].astype('uint8'))

plt.show()
datagen = ImageDataGenerator(width_shift_range=[-100,100])



it=datagen.flow_from_dataframe(df_star_wars[0:1], directory=mypath,batch_size=1)

plt.figure(figsize=(100,100))

plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)



# generate samples and plot

for i in range(9):

    plt.subplot(330 + 1 + i)

    images,labels = it.next()

    image = images[0].astype('uint8')

    plt.imshow(image)

    plt.axis('off')

# show the figure

plt.show()
datagen = ImageDataGenerator(horizontal_flip=True)

it=datagen.flow_from_dataframe(df_star_wars[0:1], directory=mypath,batch_size=1)
plt.figure(figsize=(100,100))

plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)



# generate samples and plot

for i in range(9):

    plt.subplot(330 + 1 + i)

    images,labels = it.next()

    image = images[0].astype('uint8')

    plt.imshow(image)

    plt.axis('off')

# show the figure

plt.show()
datagen = ImageDataGenerator(rotation_range=90)

it=datagen.flow_from_dataframe(df_star_wars[0:1], directory=mypath,batch_size=1)
plt.figure(figsize=(100,100))

plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)



# generate samples and plot

for i in range(9):

    plt.subplot(330 + 1 + i)

    images,labels = it.next()

    image = images[0].astype('uint8')

    plt.imshow(image)

    plt.axis('off')

# show the figure

plt.show()
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])

it=datagen.flow_from_dataframe(df_star_wars[0:1], directory=mypath,batch_size=1)
plt.figure(figsize=(100,100))

plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)



# generate samples and plot

for i in range(9):

    plt.subplot(330 + 1 + i)

    images,labels = it.next()

    image = images[0].astype('uint8')

    plt.imshow(image)

    plt.axis('off')

# show the figure

plt.show()
datagen = ImageDataGenerator(zoom_range=[0.5,1.0])

it=datagen.flow_from_dataframe(df_star_wars[0:1], directory=mypath,batch_size=1)
plt.figure(figsize=(100,100))

plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)



# generate samples and plot

for i in range(9):

    plt.subplot(330 + 1 + i)

    images,labels = it.next()

    image = images[0].astype('uint8')

    plt.imshow(image)

    plt.axis('off')

# show the figure

plt.show()