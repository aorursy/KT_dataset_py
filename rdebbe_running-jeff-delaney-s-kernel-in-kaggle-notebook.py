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



#select Theano as backend

%env KERAS_BACKEND=theano
import os, random, cv2

#import numpy as np

#import pandas as pd

from skimage.io import imread

from skimage.transform import resize

import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns

%matplotlib inline 



import keras

from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop, SGD

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils

from keras import backend as K
random.seed(2718)

K.set_image_dim_ordering('th') 

print("keras version: ", keras.__version__)

print("keras image dimension ordering: ", keras.backend.image_dim_ordering())
TRAIN_DIR = '../input/train/'

TEST_DIR =  '../input/test/'



ROWS = 64

COLS = 64

CHANNELS = 3



train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]





# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset

train_images = train_dogs[:100] + train_cats[:100]

random.shuffle(train_images)

test_images =  test_images[:25]



def read_image(file_path):

    #im = imread(file_path)

    #im = resize(im, (ROWS, COLS))

    #return im

    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)





def prep_data(images):

    count = len(images)

    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)



    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image.T

        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    

    return data



train = prep_data(train_images)

test  = prep_data(test_images)



print("Train shape: {}".format(train.shape))

print("Test shape: {}".format(test.shape))
labels = []

for i in train_images:

    if 'dog' in i:

        labels.append(1)

    else:

        labels.append(0)



sns.countplot(labels)

sns.plt.title('Cats and Dogs')
optimizer = RMSprop(lr=1e-4)



#epochs = 25

#lrate = 4.e-4

#decay = lrate/epochs

#optimizer = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)





objective = 'binary_crossentropy'





def catdog():

    

    model = Sequential()



    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS), activation='relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))



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

#model.summary()
nb_epoch = 10

batch_size = 100



## Callback for loss logging per epoch

class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = []

        self.val_losses = []

        

    def on_epoch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))



early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')        

        

def run_catdog():

    

    history = LossHistory()

    model.fit(train, labels, batch_size=batch_size, nb_epoch=nb_epoch,

              validation_split=0.25, verbose=0, shuffle=True, callbacks=[history])

    



    predictions = model.predict(test, verbose=0)

    return  predictions, history



predictions, history = run_catdog()
loss = history.losses

val_loss = history.val_losses



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('VGG-16 Loss Trend')

plt.plot(loss, 'blue', label='Training Loss')

plt.plot(val_loss, 'green', label='Validation Loss')

plt.xticks(range(0,nb_epoch)[0::2])

plt.legend()

plt.show()