# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

import keras



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

trainX = '../input/train_images'



#Adding .jpg to the image_id for later use in Keras image generator

def function(data):

    return data + '.jpg'

train['image_id'] = list(map(function, train['image_id']))
train['labels'] = train.labels.astype('str')

#Converting labels to string so they can be used by ImageGenerator
fontsize = 50



# From https://www.google.com/get/noto/

!wget -q --show-progress https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip

!unzip -p NotoSansCJKjp-hinted.zip NotoSansCJKjp-Regular.otf > NotoSansCJKjp-Regular.otf

!rm NotoSansCJKjp-hinted.zip



font = ImageFont.truetype('./NotoSansCJKjp-Regular.otf', fontsize, encoding='utf-8')
# Reference to https://www.kaggle.com/anokas/kuzushiji-visualisation for making this function 

# This function takes in a filename of an image, and the labels in the string format given in train.csv, and returns an image containing the bounding boxes and characters annotated

def visualize_training_data(image_fn, labels):

    # Convert annotation string to array

    labels = np.array(labels.split(' ')).reshape(-1, 5)

    

    # Read image

    imsource = Image.open(image_fn).convert('RGBA')

    bbox_canvas = Image.new('RGBA', imsource.size)

    char_canvas = Image.new('RGBA', imsource.size)

    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character

    char_draw = ImageDraw.Draw(char_canvas)



    for codepoint, x, y, w, h in labels:

        x, y, w, h = int(x), int(y), int(w), int(h)

        char = unicode_map[codepoint] # Convert codepoint to actual unicode character



        # Draw bounding box around character, and unicode character next to it

        bbox_draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255))

        char_draw.text((x + w + fontsize/4, y + h/2 - fontsize), char, fill=(0, 0, 255, 255), font=font)



    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)

    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

    return np.asarray(imsource)
np.random.seed(1337)

unicode_map = {codepoint: char for codepoint, char in pd.read_csv('../input/unicode_translation.csv').values}

for i in range(5):

    img, labels = train.values[np.random.randint(len(train))]

    viz = visualize_training_data('../input/train_images/{}'.format(img), labels)

    

    plt.figure(figsize=(15, 15))

    plt.title(img)

    plt.imshow(viz, interpolation='lanczos')

    plt.show()
#Creating data generator for Keras

train_datagen = ImageDataGenerator(

        rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train, directory='../input/train_images/', 

                                                    x_col='image_id', y_col='labels', target_size=(3000,2000), 

                                                    color_mode='rgb', classes=None, class_mode='categorical', batch_size=16, 

                                                    shuffle=True)