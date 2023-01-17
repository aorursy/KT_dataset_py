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

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_dir = '../input/asl_alphabet_train/asl_alphabet_train'

validation_dir = '../input/asl_alphabet_test/asl_alphabet_test'
import numpy as np

import pandas as pd

import os

import tensorflow

import keras

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras import regularizers

from tensorflow.keras.callbacks import Callback

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import clear_output
image_size = (64, 64) #Image size
datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.10, # Shift the pic width by a max of 5%

    height_shift_range=0.10, # Shift the pic height by a max of 5%

    rescale=1/255, # Rescale the image by normalzing it.

    shear_range=0.1, # Shear means cutting away part of the image (max 10%)

    zoom_range=0.1, # Zoom in by 10% max

    horizontal_flip=True, # Allow horizontal flipping

    vertical_flip=True,

    validation_split = 0.1,

    fill_mode='nearest' # Fill in missing pixels with the nearest filled value

)

train_generator = datagen.flow_from_directory( directory = '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/',

                                                     target_size = (64, 64),

                                                     class_mode = 'categorical',

                                                     batch_size = 32,

                                                     shuffle = True,

                                                     color_mode = 'rgb',

                                                     subset = 'training')
validation_generator = datagen.flow_from_directory( directory = '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/',

                                                     target_size = (64, 64),

                                                     class_mode = 'categorical',

                                                     batch_size = 32,

                                                     shuffle = True,

                                                     color_mode = 'rgb',

                                                     subset = 'validation')
print (train_generator.class_indices)



labels = '\n'.join(sorted(train_generator.class_indices.keys()))



with open('labels.txt', 'w') as f:

  f.write(labels)
class PlotLearning(Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.fig = plt.figure()

        

        self.logs = []

        



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('accuracy'))

        self.val_acc.append(logs.get('val_accuracy'))

        self.i += 1

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        

        clear_output(wait=True)

        

        ax1.set_yscale('Log')

        ax1.plot(self.x, self.losses, label="loss")

        ax1.plot(self.x, self.val_losses, label="val_loss")

        ax1.legend()

        

        ax2.plot(self.x, self.acc, label="accuracy")

        ax2.plot(self.x, self.val_acc, label="val_accuracy")

        ax2.legend()

        

        plt.show()

        

        

plot = PlotLearning()
model = Sequential()

    

model.add(Conv2D(16, kernel_size = [3,3], padding = 'same', activation = 'relu', input_shape = (64,64,3)))

model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))

model.add(MaxPool2D(pool_size = [3,3]))

    

model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))

model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'))

model.add(MaxPool2D(pool_size = [3,3]))



model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'))

model.add(Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))



model.add(Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))

model.add(Conv2D(256, kernel_size = [3,3], padding = 'same', activation = 'relu'))

model.add(MaxPool2D(pool_size = [3,3]))

    

model.add(BatchNormalization())

    

model.add(Flatten())

model.add(Dropout(0.4))



model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))

model.add(Dense(29, activation = 'sigmoid'))



model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])
model.summary()
h = model.fit_generator(train_generator, epochs=25, validation_data=validation_generator)
model.save('aslModel.h5')
plt.plot(h.history['accuracy'])

plt.plot(h.history['val_accuracy'])

plt.legend(['train', 'test'], loc='lower right')

plt.title('accuracy plot - train vs test')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.show()



plt.plot(h.history['loss'])

plt.plot(h.history['val_loss'])

plt.legend(['training loss', 'validation loss'], loc = 'upper right')

plt.title('loss plot - training vs vaidation')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()