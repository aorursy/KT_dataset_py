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
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

interpols = ["nearest","bilinear","bicubic","lanczos","box","hamming"]

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



for intepolation in interpols: 

    

    data_generator = ImageDataGenerator(

        validation_split=0.2

    )



    directory = '../input/datageneratortest2/class1'

    image = data_generator.flow_from_directory(

        directory, 

        target_size=(110, 110), 

        interpolation=intepolation

    )



    # print



    #Drawing what we have in training imgs with augmention

    x,y = image.next()

    

    fig = plt.figure(figsize=(25,45))

    for i in range(0,1):

        #print("Interpolation is :" + intepolation)

        ax = fig.add_subplot( 4, 5, i+1)

        plt.title(intepolation, y=1.08)

        image = x[i]

        ax.imshow(image.astype('uint8'))
# Orginal Image is 1578*1578 px and convert to 224*224 pixels with different image resizing method

#1562*1562





from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

interpols = ["nearest","bilinear","bicubic","lanczos","box","hamming"]

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



for intepolation in interpols: 

    

    data_generator = ImageDataGenerator(

        validation_split=0.2

    )



    directory = '../input/ourplant'

    image = data_generator.flow_from_directory(

        directory, 

        target_size=(224, 224), 

        interpolation=intepolation

    )



    # print



    #Drawing what we have in training imgs with augmention

    x,y = image.next()

    

    fig = plt.figure(figsize=(70,70))

    for i in range(0,1):

        #print("Interpolation is :" + intepolation)

        ax = fig.add_subplot( 4, 5, i+1)

        plt.title(intepolation, y=1.08)

        image = x[i]

        ax.imshow(image.astype('uint8'))
stocks = {

        'IBM': 146.48,

        'MSFT':44.11,

        'CSCO':25.54

    }



stocks['IBM']
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

interpols = ["nearest","bilinear","bicubic","lanczos","box","hamming"]

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

directory = '../input/3plants'

for intepolation in interpols: 

    

    data_generator = ImageDataGenerator(

        validation_split=0.2

    )



    image = data_generator.flow_from_directory(

        directory, 

        target_size=(222, 222), 

        interpolation=intepolation

    )



    # print



    #Drawing what we have in training imgs with augmention

    x,y = image.next()

    

    fig = plt.figure(figsize=(25,45))

    for i in range(0,3):

        #print("Interpolation is :" + intepolation)

        ax = fig.add_subplot( 4, 5, i+1)

        image = x[i]

        ax.imshow(image.astype('uint8'))