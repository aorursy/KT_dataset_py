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
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPooling2D



training_data = '../input/10-monkey-species/training/training' 

validation_data = '../input/10-monkey-species/validation/validation/'

labels_path = '../input/monkey_labels.txt'



print("Setup Done")
from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

num_classes = 10

bs = 10



def prep(image):

    img = np.array(image)

    img /= 255

    return img



aug_data_generator = ImageDataGenerator(preprocessing_function=prep,

                                    horizontal_flip = True,

                                    width_shift_range = 0.1,

                                    height_shift_range = 0.1)

data_generator = ImageDataGenerator(preprocessing_function=prep)



train_generator = aug_data_generator.flow_from_directory(

                                    directory=training_data,

                                    target_size=(image_size, image_size),

                                    batch_size=bs,

                                    class_mode='categorical')



valid_generator = aug_data_generator.flow_from_directory(

                                    directory=validation_data,

                                    target_size=(image_size, image_size),

                                    class_mode='categorical')

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu', strides=2,

                 input_shape=(image_size, image_size, 3),

                 padding='same'))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format=None))

model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation='relu', strides=1, padding='same'))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format=None))



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu', strides=2, padding='same'))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format=None))

model.add(Dropout(0.3))

model.add(Conv2D(32, (3, 3), activation='relu', strides=1, padding='same'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])
model.fit_generator(train_generator, 

                    epochs=10,

                    steps_per_epoch=1370/bs,

                   validation_data=valid_generator,

                   validation_steps=1)
model.summary()