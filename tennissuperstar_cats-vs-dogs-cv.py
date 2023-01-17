# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Necessary imports

import os, cv2, random

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns

%matplotlib inline



# Each of these imports will be explained when we build the model

from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



ROWS = 64

COLS = 64



# What are channels?  This is because we are reading the values in as RGB so 3 values/ pixel

CHANNELS = 3



# We have to do this because each image is in its own file

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]

train_dogs = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]



# Only get 1000 of each

train_images = train_dogs[:1000] + train_cats[:1000]

random.shuffle(train_images)

test_images = test_images[:25]



def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    

    # What is interpolation? This specifies that when we resize we zoom out

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)



# We store all of the data in an array of length count

def prep_data(images):

    count = len(images)

    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    

    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image.T

        

    return data



train = prep_data(train_images)



test = prep_data(test_images)

print(train.shape)
# We call the read_image function we defined in order to open the file name and display

# the pixel values of the images



def show_cats_and_dogs(idx):

    cat = read_image(train_cats[idx])

    dog = read_image(train_dogs[idx])

    pair = np.concatenate((cat, dog), axis=1)

    plt.figure(figsize=(10,15))

    plt.imshow(pair)

    plt.show()

    

for idx in range(0,5):

    show_cats_and_dogs(idx)
# How do we generate an average cat and dog picture?



avg_cat = []
# We will iterate through train and check if it is a dog and sum over that way

# Train has already been processed through read_image which reads in using

# cv.imread()



# We want to be able to iterate through and figure out which is a cat and which is 

# a dog so let's create an array of labels



labels = []

for img in train_images:

    if 'dog' in img:

        labels.append(1)

    else:

        labels.append(0)



# We store 3 channels and thus need to get dog[0], this is the first channel



dog_avg = np.array([dog[0].T for index, dog in enumerate(train) 

                    if labels[index] == 1]).mean(axis=0)

print(dog_avg.shape)

# Now I want to display my avg_dog

plt.imshow(dog_avg)

plt.show()

# Now let me create an average cat

cat_avg = np.array([cat[0].T for i, cat in enumerate(train) 

                    if labels[i] == 0]).mean(axis=0)

plt.imshow(cat_avg)

plt.show()
optimizer = RMSprop (lr=1e-4)

objective = 'binary_crossentropy'



def catdog():

    # How did you come up with this layer technique?

    model = Sequential()

    

    model.add(Convolution2D(32, 3, 3, border_mode='same', 

                            input_shape=(3, ROWS, COLS), activation='relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same', 

                           activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))

    

    model.add(Convolution2D(64, 3, 3, border_mode='same', 

                            input_shape=(3, ROWS, COLS), activation='relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', 

                           activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))

    

    model.add(Convolution2D(128, 3, 3, border_mode='same', 

                            input_shape=(3, ROWS, COLS), activation='relu'))

    model.add(Convolution2D(128, 3, 3, border_mode='same', 

                           activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))

    

    model.add(Convolution2D(256, 3, 3, border_mode='same', 

                            input_shape=(3, ROWS, COLS), activation='relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', 

                           activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))

    

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    return model



model = catdog()
nb_epoch = 10

batch_size = 16



class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = []

        self.val_losses = []

        

    def on_epoch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))