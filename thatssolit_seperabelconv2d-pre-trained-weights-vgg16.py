# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('../input/flowers-recognition/flowers'))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob

from pathlib import Path

flower_dir = Path('../input/flowers-recognition/flowers/*/*')

flowers = {}

for directory in glob.glob(os.path.abspath(flower_dir)):

    dir_list = []

    dir_list = directory.split('/')

    images = []

    for files in glob.glob(directory + '/*.jpg'):

        images.append(files)

        flowers.setdefault(dir_list[-1], images)

    

label = []

Image = []

for keys, values in flowers.items():

    for v in values:

        Image.append(v)

        label.append(keys)



len(Image), len(label)
from sklearn.utils import shuffle

import cv2

import matplotlib.pyplot as plt



Image, label = shuffle(Image, label, random_state = 0)



f, ax = plt.subplots(2,5, figsize = (16,8))

for i in range(10):

    img = Image[i]

    img = cv2.imread(img)

    img = cv2.resize(img, (224,224))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax[i//5, i%5].imshow(img)

    ax[i//5, i%5].axis('off')

    ax[i//5, i%5].set_title(label[i])

plt.show()
data_df = pd.DataFrame(list(zip(Image, label)),columns = ['Image', 'Label'])

data_df
types_of_flowers = ['rose','dandelion', 'daisy', 'tulip', 'sunflower']

map = {}

for i in range(len(types_of_flowers)):

    map[types_of_flowers[i]] = i





for x in range(data_df.shape[0]):

    data_df['Label'][x] = map[data_df['Label'][x]]

data_df
from keras.utils import to_categorical



def data_generator(data, batch_size):

    num_samples = len(data)

    while True:

        for offset in range(0, num_samples, batch_size):

            batch_samples = data[offset: offset+batch_size]

            X = []

            y = []

            for ind in batch_samples.index:

                img = batch_samples['Image'][ind]

                label = batch_samples['Label'][ind]

                

                encoded_label = to_categorical(label, num_classes=5)

                

                img = cv2.imread(str(img))

                img = cv2.resize(img, (224,224))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                

                if img.shape[2]==1:

                    img = np.dstack([img, img, img])

                img = img.astype(np.float32)/255    

                

                X.append(img)

                y.append(encoded_label)

            

            

            X = np.asarray(X).astype(np.float32)

            y = np.asarray(y)

            

            yield X, y

        

        

    
                                              

import tensorflow as tf

from keras.layers import Conv2D, Flatten, Dense, Dropout, Input, MaxPooling2D, GlobalMaxPooling2D, SeparableConv2D

from keras.layers.normalization import BatchNormalization

from keras.models import Model

from keras.optimizers import Adam, SGD, RMSprop



def build_model():

    input_img = Input(shape = (224,224,3), name = 'ImageInput')

    x = Conv2D(64, (3,3), activation = 'relu', padding = 'same', name = 'Conv1_1' )(input_img)

    x = Conv2D(63, (3,3), activation = 'relu', padding = 'same', name = 'Conv1_2' )(x)

    x = MaxPooling2D((2,2), name = 'pool1')(x)

    

    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation = 'relu', padding = 'same', name = 'Conv2_1')(x)

    x = SeparableConv2D(128, (3,3), activation = 'relu', padding = 'same', name = 'Conv2_2')(x)

    x = MaxPooling2D((2,2), name = 'pool2')(x)

    

    x = tf.keras.layers.SeparableConv2D(256, (3,3), activation = 'relu', padding = 'same', name = 'Conv3_1')(x)

    x = BatchNormalization(name = 'bn1')(x)

    x = tf.keras.layers.SeparableConv2D(256, (3,3), activation = 'relu', padding = 'same', name = 'Conv3_2')(x)

    x = BatchNormalization(name = 'bn2')(x)

    x = tf.keras.layers.SeparableConv2D(256, (3,3), activation = 'relu', padding = 'same', name = 'Conv3_3')(x)

    x = MaxPooling2D((2,2), name = 'pool3')(x)

    

    x = tf.keras.layers.SeparableConv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'Conv4_1')(x)

    x = BatchNormalization(name = 'bn3')(x)

    x = tf.keras.layers.SeparableConv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'Conv4_2')(x)

    x = BatchNormalization(name = 'bn4')(x)

    x = tf.keras.layers.SeparableConv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'Conv4_3')(x)

    x = MaxPooling2D((2,2), name = 'pool4')(x)

    

    x = Flatten(name = 'flatten')(x)

    x = Dense(1024,activation = 'relu', name = 'fc1')(x)

    x = Dropout(0.7, name = 'dropout1')(x)

    x = Dense(512, activation = 'relu', name = 'fc2')(x)

    x = Dropout(0.5, name = 'dropout2')(x)

    x = Dense(5, activation = 'softmax', name = 'fc3')(x)

    

    model = Model(inputs = input_img, outputs = x)

    return model
model = build_model()

model.summary()


from keras.applications.vgg16 import VGG16, preprocess_input

import h5py



f = h5py.File('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')



w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']

model.layers[1].set_weights = [w,b]



w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']

model.layers[2].set_weights = [w,b]



w,b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']

model.layers[4].set_weights = [w,b]

w,b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']

model.layers[5].set_weights = [w,b]



f.close()

model.summary()
batch_size = 20

opt = Adam(lr=0.0001, decay=1e-5)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

from sklearn.model_selection import train_test_split



X_train, X_val = train_test_split(data_df, test_size = 0.25, random_state = 30)





print(X_train.shape)

print(X_val.shape)

batch_size = 20

train_data_gen = data_generator(X_train, batch_size)

val_data_gen = data_generator(X_val, batch_size)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



earlyStopping = EarlyStopping(patience = 10)

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 10, verbose =1, factor = 0.5,

                                           min_lr = 0.0001)

callback = [earlyStopping, learning_rate_reduction]



ephs = 20

steps_per_ephs = X_train.shape[0]//batch_size

val_steps = X_val.shape[0]//batch_size
history = model.fit_generator(train_data_gen, epochs = ephs, validation_data = val_data_gen,

                             validation_steps = val_steps, steps_per_epoch = steps_per_ephs,

                             callbacks = callback)
f, (ax1, ax2) = plt.subplots(2,1, figsize = (12,12))

ax1.plot(history.history['accuracy'], label = 'Training Accuracy')

ax1.plot(history.history['val_accuracy'], label = 'Validation Accuracy')

ax1.set_title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

ax1.legend(['Training Data', 'Validation Data'], loc = 'best')



ax2.plot(history.history['loss'], label = 'Training loss')

ax2.plot(history.history['val_loss'], label = 'Validation Loss')

ax2.set_title('Model Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

ax2.legend(['Training Data', 'Validation Data'], loc = 'best')