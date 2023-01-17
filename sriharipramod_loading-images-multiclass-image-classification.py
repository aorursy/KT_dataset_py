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
import matplotlib.pyplot as plt

%matplotlib inline



#Import ImageDataGenerator for Loading Images

from keras.preprocessing.image import ImageDataGenerator
# We specify image augmentation parameters as the arguments

# Train - test/validation split can be done with the argument - validation_split

datagen = ImageDataGenerator(rescale=1./255,

                            validation_split = 0.1,rotation_range=30,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True,

                                    fill_mode='nearest')

# flow_from_directory gets label for an image from the sub-directory it is placed in

# Generate Train data

traingenerator = datagen.flow_from_directory(

        '../input/10_categories-1563192636507/10_categories',

        target_size=(75, 75),

        batch_size=3350,

        subset='training',

        class_mode='categorical')



# Generate Validation data

valgenerator = datagen.flow_from_directory(

        '../input/10_categories-1563192636507/10_categories',

        target_size=(75, 75),

        batch_size=360,

        subset='validation',

        class_mode='categorical')
x_train,y_train = next(traingenerator)



x_test,y_test = next(valgenerator)
plt.figure(figsize=(20,10))

for i in range(6):

    plt.subplot(1,6,i+1)

    plt.imshow(x_train[i])