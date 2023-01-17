from tensorflow.python.framework import ops

ops.reset_default_graph()

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow.keras

from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import os

print(os.listdir("../input/filteredemnist"))
train_data_path = '../input/filteredemnist/filtered-emnist-train.csv'

test_data_path = '../input/filteredemnist/filtered-emnist-test.csv'
train_data = pd.read_csv(train_data_path, header=None)
train_data.head(10)
# The classes of this balanced dataset are as follows. Index into it based on class label

class_mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

# source data: https://arxiv.org/pdf/1702.05373.pdf
class_mapping[34]
train_data.shape
num_classes = len(train_data[0].unique())

row_num = 8

img_flip = np.transpose(train_data.values[row_num,1:].reshape(28, 28), axes=[1,0]) # img_size * img_size arrays

# 10 digits, 26 letters, and 11 capital letters that are different looking from their lowercase counterparts

num_classes = 37 

img_size = 28



def img_label_load(data_path, num_classes=None):

    data = pd.read_csv(data_path, header=None)

    print(data.head(5))

    data_rows = len(data)

    if not num_classes:

        num_classes = len(data[0].unique())

    

    # this assumes square imgs. Should be 28x28

    img_size = int(np.sqrt(len(data.iloc[0][1:])))

    

    # As emnist data is in wrong shape/tilted. 

    # Images need to be transposed. This line also does the reshaping needed.

    imgs = np.transpose(data.values[:,1:].reshape(data_rows, img_size, img_size, 1), axes=[0,2,1,3]) # img_size * img_size arrays

    

    labels = keras.utils.to_categorical(data.values[:,0], num_classes) # one-hot encoding vectors

    

    return imgs/255., labels



# Define the optimizer

from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model = tensorflow.keras.models.Sequential()

model.add(Conv2D(32, kernel_size=(5, 5),

                 activation='relu',

                 input_shape=(img_size,img_size,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()
for layer in model.layers:

    print(layer.get_output_at(0).get_shape().as_list())

X, y = img_label_load(train_data_path)

print(X.shape)

print(y)

# Set a learning rate annealer

from tensorflow.keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
# Data generator add randomness to existing data.

data_generator = keras.preprocessing.image.ImageDataGenerator(validation_split=.2)

## consider using this for more variety

data_generator_with_aug = keras.preprocessing.image.ImageDataGenerator(validation_split=.2, width_shift_range=.1,

                                                                       height_shift_range=.1, rotation_range=20,

                                                                       zoom_range=.1, shear_range=.1)



# if already ran this above, no need to do it again

# X, y = img_label_load(train_data_path)

# print("X.shape: ", X.shape)



training_data_generator = data_generator_with_aug.flow(X, y, subset='training')

validation_data_generator = data_generator_with_aug.flow(X, y, subset='validation')

history = model.fit_generator(training_data_generator, 

                              steps_per_epoch=500, epochs=30,

                              validation_data=validation_data_generator, 

                              validation_steps= len(X) / 500,

                              callbacks=[learning_rate_reduction])
test_X, test_y = img_label_load(test_data_path)

test_data_generator = data_generator.flow(X, y)



model.evaluate_generator(test_data_generator)
model.save('./alphabet_a_z.h5')
from IPython.display import FileLink, FileLinks

FileLinks('.') #lists all downloadable files on server