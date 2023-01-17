# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # showing and rendering figures

from skimage.io import imread

from glob import glob

#from IPython.display import Image as Image1 # show image real-time

from PIL import Image as Image # convert image to numpy array



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import sys



%matplotlib inline

# Any results you write to the current directory are saved as output.

    
from keras.layers import Input, Dense, Activation, Conv2D, BatchNormalization

from keras.layers import Flatten, MaxPooling2D, Dropout, Conv2DTranspose

from keras.layers import Concatenate, Lambda

from keras.models import Model

from keras.utils import to_categorical

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import cv2

import tensorflow as tf

import os

# read data

IMG_SIZE = 512





train_data = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

print('{} samples'.format(len(train_data)))



test_data = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

print('{} tests'.format(len(test_data)))



train_ids = train_data['id_code'].values

train_labels = train_data['diagnosis'].values



test_ids = test_data['id_code'].values







TRAIN_PATH = '../input/aptos2019-blindness-detection/train_images'

TEST_PATH = '../input/aptos2019-blindness-detection/test_images'



"""

Arguments:

    num_batch: start from 1

"""

def batch_read(num_batch, batch_size=100):

    x_train, y_train = [], []

    for i in tqdm(range(batch_size)):

        img_file = os.path.join(TRAIN_PATH, train_ids[i+batch_size*(num_batch-1)]+'.png')

        img = cv2.imread(img_file)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img.astype('float32')/255

        x_train.append(img)

        

        label = train_labels[i+batch_size*(num_batch-1)]

        label = to_categorical(label, num_classes=5)

        y_train.append(label)

        

    return np.array(x_train), np.array(y_train)



x_train, y_train = batch_read(1)

# randomly show 15 images



plt.figure(figsize=(20, 15))

idx = np.random.randint(0, len(x_train), size=15)



for i in range(15):

    plt.subplot(3, 5, i+1)

    im = x_train[idx[i]][..., [2, 1, 0]] # tranverse RGB to 

    im = plt.imshow(im)

    plt.title('image %s label %s' % (i+1, np.argmax(y_train[idx[i]])))

plt.show()

print(im.get_cmap().name)
# show label distrubtion

import seaborn as sns

plt.figure(figsize=(20, 10))

sns.set(style='darkgrid')

ax= sns.countplot(x='diagnosis', data=train_data, palette="GnBu_d") # accept pd.DataFrame

plt.show()
# get ground truth 

file = os.path.join(TRAIN_PATH, train_ids[0]+'.png')

img = cv2.imread(file, 0)
# customized metric.accuracy

def mean_iou(y_true, y_pred):

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):

        y_pred_ = tf.to_int32(y_pred > t) # tensor 1 / 0

        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)

        K.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([up_opt]):

            score = tf.identity(score) # no change

        prec.append(score)

    return K.mean(K.stack(prec), axis=0)
# use mnist to test model

mnist = pd.read_csv('../input/digit-recognizer/train.csv')

labels = mnist['label'].values

m_images = mnist.drop(['label'], axis=1).values.reshape(-1, 28, 28, 1)

m_labels = to_categorical(labels, num_classes=10)
# Unet in paper

# dimension needs to be resized by tf.image.resize_images

'''

input = Input(shape=(image_size, image_size, 3))

input = cv2.resize(a_images[0], (image_size, image_size))

input = input.reshape(1, image_size, image_size, 3)

input = tf.Variable(input)



## Contracting

# 1 - 64

conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))(input) # (?, 570, 570, 64)

conv1 = BatchNormalization()(conv1)

conv1 = Activation('relu')(conv1)



conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))(conv1) # (?, 568, 568, 64)

conv1 = BatchNormalization()(conv1)

conv1 = Activation('relu')(conv1)



pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1) # (?, 284, 284, 64)





# 2 - 128

conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(pool1) # (?, 282, 282, 128)

conv2 = BatchNormalization()(conv2)

conv2 = Activation('relu')(conv2)



conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(conv2) # (?, 280, 280, 128)

conv2 = BatchNormalization()(conv2)

conv2 = Activation('relu')(conv2)



pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2) # (?, 140, 140, 128)





# 3 - 256

conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1))(pool2) # (?, 138, 138, 256)

conv3 = BatchNormalization()(conv3)

conv3 = Activation('relu')(conv3)



conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1))(conv3) # (?, 136, 136, 256)

conv3 = BatchNormalization()(conv3)

conv3 = Activation('relu')(conv3)



pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3) # (?, 68, 68, 256)





# 4 - 512

conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1))(pool3) # (?, 66, 66, 512)

conv4 = BatchNormalization()(conv4)

conv4 = Activation('relu')(conv4)



conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1))(conv4) # (?, 64, 64, 512)

conv4 = BatchNormalization()(conv4)

conv4 = Activation('relu')(conv4)



pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4) # (?, 32, 32, 512)





# 5 - 1024

conv5 = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1))(pool4) # (?, 30, 30, 1024)

conv5 = BatchNormalization()(conv5)

conv5 = Activation('relu')(conv5)



conv5 = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1))(conv5) # (?, 28, 28, 1024)

conv5 = BatchNormalization()(conv5)

conv5 = Activation('relu')(conv5)





## Expansive

# 1 - 512

dconv1 = Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2))(conv5)  #(56, 56, 512)

dconv1 = BatchNormalization()(dconv1)

dconv1 = Activation('relu')(dconv1)



cat1 = Concatenate(axis=3)([conv4, dconv1]) # (56, 56, 1024)



conv6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1))(cat1) # (54, 54, 512)

conv6 = BatchNormalization()(conv6)

conv6 = Activation('relu')(conv6)



conv6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1))(conv6) # (52, 52, 512)

conv6 = BatchNormalization()(conv6)

conv6 = Activation('relu')(conv6)



se=tf.Session()

se.run(tf.global_variables_initializer())

result = se.run(dconv1)

print(result.shape)

'''
# Unet in paper (revised)

# 1) image_size should be 32n(5 layers)

# 2) Conv2D use padding='same' to avoid concatenate problem



image_size = 512



input = Input(shape=(image_size, image_size, 3))



## Contracting

# 1 - 64

conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input) # (?, 512, 512, 64)

conv1 = BatchNormalization()(conv1)

conv1 = Activation('relu')(conv1)



conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1) # (?, 512, 512, 64)

conv1 = BatchNormalization()(conv1)

conv1 = Activation('relu')(conv1)



pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1) # (?, 256, 256, 64)





# 2 - 128

conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool1) # (?, 256, 256, 128)

conv2 = BatchNormalization()(conv2)

conv2 = Activation('relu')(conv2)



conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv2) # (?, 256, 256, 128)

conv2 = BatchNormalization()(conv2)

conv2 = Activation('relu')(conv2)



pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2) # (?, 128, 128, 128)





# 3 - 256

conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool2) # (?, 128, 128, 256)

conv3 = BatchNormalization()(conv3)

conv3 = Activation('relu')(conv3)



conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv3) # (?, 128, 128, 256)

conv3 = BatchNormalization()(conv3)

conv3 = Activation('relu')(conv3)



pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3) # (?, 64, 64, 256)





# 4 - 512

conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool3) # (?, 64, 64, 512)

conv4 = BatchNormalization()(conv4)

conv4 = Activation('relu')(conv4)



conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv4) # (?, 64, 64, 512)

conv4 = BatchNormalization()(conv4)

conv4 = Activation('relu')(conv4)



pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4) # (?, 32, 32, 512)





# 5 - 1024

conv5 = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool4) # (?, 32, 32, 1024)

conv5 = BatchNormalization()(conv5)

conv5 = Activation('relu')(conv5)



conv5 = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv5) # (?, 32, 32, 1024)

conv5 = BatchNormalization()(conv5)

conv5 = Activation('relu')(conv5)





## Expansive

# 1 - 512

dconv1 = Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2))(conv5)  #(64, 64, 512)

dconv1 = BatchNormalization()(dconv1)

dconv1 = Activation('relu')(dconv1)



cat1 = Concatenate(axis=3)([conv4, dconv1]) # (64, 64, 1024)



conv6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(cat1) # (64, 64, 512)

conv6 = BatchNormalization()(conv6)

conv6 = Activation('relu')(conv6)



conv6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv6) # (64, 64, 512)

conv6 = BatchNormalization()(conv6)

conv6 = Activation('relu')(conv6)





# 2 - 256

dconv2 = Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=(2, 2))(conv6) # (128, 128, 256)

dconv2 = BatchNormalization()(dconv2)

dconv2 = Activation('relu')(dconv2)



cat2 = Concatenate(axis=3)([conv3, dconv2]) # (128, 128, 512)



conv7 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(cat2) # (128, 128, 256)

conv7 = BatchNormalization()(conv7)

conv7 = Activation('relu')(conv7)



conv7 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv7) # (128, 128, 256)

conv7 = BatchNormalization()(conv7)

conv7 = Activation('relu')(conv7)





# 3 - 128

dconv3= Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2))(conv7) # (256, 256, 128)

dconv3 = BatchNormalization()(dconv3)

dconv3 = Activation('relu')(dconv3)



cat3 = Concatenate(axis=3)([conv2, dconv3]) # (256, 256, 256)



conv8 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(cat3) # (256, 256, 128)

conv8 = BatchNormalization()(conv8)

conv8 = Activation('relu')(conv8)



conv9 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv8) # (256, 256, 128)

conv9 = BatchNormalization()(conv9)

conv9 = Activation('relu')(conv9)





# 4 - 64

dconv4 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2))(conv9) # (512, 512, 64)

dconv4 = BatchNormalization()(dconv4)

dconv4 = Activation('relu')(dconv4)



cat4 = Concatenate(axis=3)([conv1, dconv4]) # (512, 512, 128)



conv10 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(cat4) # (512, 512, 64)

conv10 = BatchNormalization()(conv10)

conv10 = Activation('relu')(conv10)



conv11 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv10) # (512, 512, 64)

conv11 = BatchNormalization()(conv11)

conv11 = Activation('relu')(conv11)



output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv11) # (512, 512, 1)



print(output)



model  = Model(input, output)

model.compile(loss='categorical_crossentropy', 

              optimizer='adam',

              metrics=mean_iou)

model.summary()

%%time

from keras.callbacks import EarlyStopping, ModelCheckpoint



earlystopping = EarlyStopping(patience=5, verbose=1)

checkpoint = ModelCheckpoint('model-aptos.h5', verbose=1, save_best_only=True)



history = model.fit(a_images, a_labels, epochs=30, batch_size=2, validation_split=0.1, callbacks=[earlystopping, checkpoint])



plt.plot(history.history['acc'])

plt.title('Accuracy')

plt.show()



plt.plot(history.history['loss'])

plt.title('Loss')

plt.show()
from tqdm import tnrange, tqdm_notebook

from time import sleep



for i in tnrange(10, desc='1st loop'):

    sleep(1)

    

for j in tqdm_notebook(range(100), desc='2nd loop'):

    sleep(0.01)