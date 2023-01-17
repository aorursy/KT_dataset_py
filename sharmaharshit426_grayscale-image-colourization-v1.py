# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input/gray-colourize/data/Test/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# Any results you write to the current directory are saved as output.
from IPython.display import display, Image

from skimage.color import rgb2lab, lab2rgb

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D, UpSampling2D, InputLayer, MaxPooling2D

from keras.callbacks import TensorBoard

import datetime

import h5py

from keras.utils import plot_model

import matplotlib.pyplot as plt

'''

%load_ext tensorboard

%tensorboard --logdir logs

'''
def create_dataset(data_path, m, img_h, img_w):

    

    # path  = Path of the data set. Number of training examples(m) will be equal to number of files in the specified path

    # m     = Number of training Examples

    # img_h = Height of the image(Input images will be resized to have this number of rows)

    # img_w = Width of the image(Input images will be resized to have this number of columns)

     

    dataset = np.ndarray(shape=(m, img_h, img_h, 1))

    Y = np.ndarray(shape=(m, img_h, img_h, 2))

    

    #Data set is of the dimensions (m, number_of_channels, no_of_rows, No_of_columns)

    i = 0

    for dirname, _, filenames in os.walk(data_path):

        for file_name in filenames:

            img = load_img(os.path.join(dirname, file_name) ,target_size = (img_h, img_w))

            x = img_to_array(img)/255

            lab_image = rgb2lab(x)

            lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]

            x = lab_image[:,:,0]

            y = lab_image[:,:,1:]



            #x = x.reshape(1, img_h, img_w)

            #y = y.reshape(2, img_h, img_w)

            x = x.reshape(img_h, img_w, 1)

            y = y.reshape(img_h, img_w, 2)

            dataset[i] = x

            Y[i] = y

            i += 1

            if i % 200 == 0:

                print(f"{i} images added to array")

    print("All images have been added to array!")

    return dataset, Y
train_path = '/kaggle/input/gray-colourize/data/Training/'

m_train = 6070

img_h = 200

img_w = 200

# Creating our dataset

X_train, Y_train = create_dataset(data_path=train_path, m=m_train, img_h=img_h, img_w=img_w)

print(f'Shape of X_train is {X_train.shape}\nShape of Y_train is {Y_train.shape}')



test_path = '//kaggle/input/gray-colourize/data/Test'

m_test = 310

X_test, Y_test = create_dataset(data_path=test_path, m=m_test, img_h=img_h, img_w=img_w)

print(f'Shape of X_test is {X_test.shape}\nShape of Y_test is {Y_test.shape}')
# detect and init the TPU



tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# Creating our Model

with tpu_strategy.scope():

    model = Sequential()

    model.add(InputLayer(input_shape=(200, 200, 1)))

    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))

    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))

    model.summary()

    model.compile(optimizer='Adam', loss='mean_squared_logarithmic_error' ,metrics=['accuracy'])



'''

model.add(Conv2D(filters=3, kernel_size=(3, 3), activation='relu', padding='same', use_bias=True, strides=1))

model.add(MaxPooling2D(pool_size=(2*2)))

model.add(UpSampling2D(size=(2,2)))

model.add(UpSampling2D(size=(2,2)))



model.add(Conv2D(filters=6, kernel_size=(3,3), activation='tanh', padding='same', use_bias=True, strides=2))

model.add(MaxPooling2D(pool_size=(2*2)))

model.add(UpSampling2D(size=(2,2)))

model.add(UpSampling2D(size=(2,2)))



model.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', use_bias=True, strides=1))



model.add(Conv2D(filters=16, kernel_size=(3,3), activation='tanh', padding='same', use_bias=True, strides=2))

model.add(UpSampling2D(size=(2,2)))



model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', use_bias=True, strides=1))



model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', use_bias=True, strides=1))

model.add(UpSampling2D(size=(2,2)))



model.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', use_bias=True, strides=1))



model.add(Conv2D(filters=2, kernel_size=(3,3), activation='relu', padding='same', use_bias=True, strides=2))

model.add(UpSampling2D(size=(2,2)))



model.summary()

model.compile(optimizer='Adam', loss='mean_squared_logarithmic_error' ,metrics=['accuracy'])

'''

plot_model(model, show_shapes=True, show_layer_names=True ,to_file='model.png')

#logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#tensorboard_callback = TensorBoard("logs", histogram_freq=1)

#model.load_weights('/kaggle/input/weights75/my_model75.h5')
history = model.fit(x=X_train, y=Y_train, verbose=1, epochs=100, validation_data=(X_test, Y_test))

# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
test1 = load_img('/kaggle/input/cars-colored/car_set/car_0086.jpg', target_size=(200,200))

test1_arr = img_to_array(test1)/255

lab_image = rgb2lab(test1_arr)

#lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]

x = lab_image[:,:,0]

y = lab_image[:,:,1:]

x = x.reshape(img_h, img_w, 1)

y = y.reshape(img_h, img_w, 2)



x = x.reshape(1, img_h, img_w, 1)

y = y.reshape(1, img_h, img_w, 2)



output = model.predict(x)

cur = np.zeros((200, 200, 3))

cur[:,:,0] = x[0][:,:,0]

cur[:,:,1:] = output[0]

#cur = (cur * [100, 255, 255]) - [0, 128, 128]

rgb_image = lab2rgb(cur)



fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(test1)

ax2.imshow(rgb_image)
y.shape
model.save('my_model150_6k.h5')

model.save_weights('my_model_weights150_6k.h5')
print('yes')
model1 = model.load_weights('my_model.h5')
print('okay boomer')

#this is just some random stuff to keep the notbook alive
#what's up guys?

hello
#whats up helooooooodsfsdfdsfdsf

asdsadsadasd