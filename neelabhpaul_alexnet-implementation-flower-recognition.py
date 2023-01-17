import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from tensorflow import keras

from sklearn.preprocessing import LabelBinarizer

import cv2



tf.config.experimental.list_physical_devices('GPU') 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)
# Check for available GPU

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

    raise SystemError('GPU not found')

print('Found GPU at: {}'.format(device_name))
from PIL import Image        # for image processing

with tf.device('/GPU:0'):

    image_set = []

    label_set = []

    path = '../input/flowers-recognition/flowers'

    for flower_type in os.listdir(path):

        subpath = os.path.join(path, flower_type)

        for img in os.listdir(subpath):

            try:

                flower_pic = os.path.join(subpath,img)

                image = cv2.imread(flower_pic)

                image = cv2.resize(image, (224,224))

                image_set.append(image)

                label_set.append(flower_type)

            except Exception as e:           # To remove problematic pictures and prevent the program from encountering errors

                print(str(e))
print(len(label_set))



print(label_set[769])

plt.imshow(image_set[769])
image_set = np.array(image_set)

label_set = pd.Series(label_set)

image_set.shape
label_set.shape
label_set.head()
label_set.unique()
label_set = label_set.map({'daisy':1, 'sunflower':2, 'tulip':3, 'rose':4, 'dandelion':5})

label_set.head()
label_set = pd.DataFrame(label_set) # to convert the shape (4323,) to (4323,1)

label_set.shape
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(image_set, label_set, test_size=0.2, random_state=37)

print(len(train_x), len(train_y), len(test_x), len(test_y))
# One vs all classification

label_binrizer = LabelBinarizer()

train_y = label_binrizer.fit_transform(train_y)
image_set[0].shape
from keras.layers import Concatenate
def GoogLeNet():

    

    def inception(x, f):

        t1 = tf.keras.layers.Conv2D(f[0], 1, activation='relu')(x)



        t2 = tf.keras.layers.Conv2D(f[1], 1, activation='relu')(x)

        t2 = tf.keras.layers.Conv2D(f[2], 3, padding='same', activation='relu')(t2)



        t3 = tf.keras.layers.Conv2D(f[3], 1, activation='relu')(x)

        t3 = tf.keras.layers.Conv2D(f[4], 5, padding='same', activation='relu')(t3)



        t4 = tf.keras.layers.MaxPooling2D(3, 1, padding='same')(x)

        t4 = tf.keras.layers.Conv2D(f[5], 1, activation='relu')(t4)



        layer = Concatenate()([t1, t2, t3, t4])

    

        return layer



    with tf.device(device_name):

        class myCallback(tf.keras.callbacks.Callback):        # interrupts the training when 99.9% is achieved

            def on_epoch_end(self, epoch, logs={}):

                if(float(logs.get('accuracy'))>0.999):

                    print("\nReached 99.9% accuracy so cancelling training!")

                    self.model.stop_training = True

                

        callbacks = myCallback()

        

        input = tf.keras.Input([224, 224, 3])

        # Layer1

        x = tf.keras.layers.Conv2D(64,(7,7),strides=(2,2),padding='same',activation=tf.nn.relu)(input)

        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

        # Layer2

        x = tf.keras.layers.Conv2D(192,(3,3),strides=(1,1),padding='same',activation=tf.nn.relu)(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

        # Layer3

        x = inception(x, [64, 96, 128, 16, 32, 32])

        x = inception(x, [128, 128, 192, 32, 96, 64])

        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

        # Layer4

        x = inception(x, [192, 96, 208, 16, 48, 64])

        x = inception(x, [160, 112, 224, 24, 64, 64])

        x = inception(x, [128, 128, 256, 24, 64, 64])

        x = inception(x, [112, 144, 288, 32, 64, 64])

        x = inception(x, [256, 160, 320, 32, 128, 128])

        x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

        # Layer5

        x = inception(x, [256, 160, 320, 32, 128, 128])

        x = inception(x, [384, 192, 384, 48, 128, 128])

        x = tf.keras.layers.AveragePooling2D(6,strides=1)(x)

        x = tf.keras.layers.Dropout(0.4)(x)

        output = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)

        

        model = tf.keras.Model(input, output)

    return model
def AlexNet(images, labels):

    with tf.device(device_name):

        class myCallback(tf.keras.callbacks.Callback):        # interrupts the training when 99.9% is achieved

            def on_epoch_end(self, epoch, logs={}):

                if(float(logs.get('accuracy'))>0.999):

                    print("\nReached 99.9% accuracy so cancelling training!")

                    self.model.stop_training = True

                

        callbacks = myCallback()

        model = tf.keras.models.Sequential([

            # Layer 1

            tf.keras.layers.Conv2D(96,(11,11), strides=4, padding='same', activation=tf.nn.relu),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.MaxPooling2D(3,2),

            # Layer 2

            tf.keras.layers.Conv2D(256,(5,5), strides=2, padding='same', activation=tf.nn.relu),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.MaxPooling2D(3,2),

            # Layer 3

            tf.keras.layers.Conv2D(384,(3,3), strides=1, padding='same', activation=tf.nn.relu),

            # Layer 4

            tf.keras.layers.Conv2D(384,(3,3), strides=1, padding='same', activation=tf.nn.relu),

            # Layer 5

            tf.keras.layers.Conv2D(256,(3,3), strides=1, padding='same', activation=tf.nn.relu),

            tf.keras.layers.MaxPooling2D(3,2),



            # FC 1

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(4096, activation=tf.nn.relu),

            # FC 2

            tf.keras.layers.Dense(4096, activation=tf.nn.relu),

            # Softmax layer

            tf.keras.layers.Dense(5, activation=tf.nn.softmax)



        ])



        model.compile(keras.optimizers.Adam(learning_rate=0.0001), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

        hist = model.fit(images, labels, epochs=50, callbacks=[callbacks])

    return model
train_x = train_x/255

test_x = test_x/255

model = AlexNet(train_x, train_y)

model.summary()
from sklearn.metrics import accuracy_score

pred_y = model.predict(test_x)

# One vs all classification

test_y = label_binrizer.fit_transform(test_y)

print('AlexNet test accuracy: ',accuracy_score(test_y, pred_y.round()))