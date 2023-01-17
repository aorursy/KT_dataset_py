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
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.python.framework import ops
import math
import glob
import zipfile
import tensorflow_datasets as tfds
# from skimage.transform import resize   # for resizing images

AUTOTUNE = tf.data.experimental.AUTOTUNE
# constants:
BATCH_SIZE=64
#BATCH_SIZE=56
target_image_dim = 224
#target_image_dim = 336
local_zip = '/kaggle/input/cassava-disease/train.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
trainDIR = '/tmp/train/'
valDIR = '/tmp/val/'
testDIR = '/tmp/test/'
if (not(os.path.exists(valDIR))):
    os.mkdir(valDIR)
    for cx in os.listdir(trainDIR):
        os.mkdir(os.path.join(valDIR,cx))
if (not(os.path.exists(testDIR))):
    os.mkdir(testDIR)
    for cx in os.listdir(trainDIR):
        os.mkdir(os.path.join(testDIR,cx))
cxnames = os.listdir(trainDIR)
numSamples = [len(os.listdir(os.path.join(trainDIR,c)))  for c in cxnames]
numClasses = len(cxnames)
for j in range(numClasses):
    print('class %s: %d/%d' % (cxnames[j],numSamples[j],sum(numSamples)))

#Actually let's grab the data from TFDS 
cassava, info =  tfds.load('cassava', as_supervised=True, with_info=True)
info
train_dataset,val_dataset,test_dataset = cassava['train'], cassava['validation'], cassava['test']
num_train_examples= info.splits['train'].num_examples
num_val_examples= info.splits['validation'].num_examples
num_test_examples= info.splits['test'].num_examples
print((num_train_examples,num_val_examples,num_test_examples))
# Same number of training images as Kaggle ds
def convert_and_resize(image,label):
    image = tf.image.convert_image_dtype(image,tf.float32) # cast and normalize the image to [0,1]
    image = tf.image.resize_with_pad(image, target_image_dim, target_image_dim)
    return (image,label)
    
def augment(image,label):
    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.resize_with_pad(image, target_image_dim, target_image_dim)
    image = tf.image.random_crop(image,size=(target_image_dim//2,target_image_dim//2,3))
    image = tf.image.resize_with_pad(image, target_image_dim, target_image_dim)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image,0.0,3.0)
    image = tf.image.random_brightness(image,0.3)
    image = tf.image.random_contrast(image,0.5,2.0)
    return (image,label)
def killImage(image,label):
    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.resize_with_pad(image, target_image_dim, target_image_dim)
    image =  tf.image.adjust_brightness(image,-2)
    return (image,label)
# Build BATCHES:
augmented_train_batches = (
    train_dataset
    .cache()
    .shuffle(num_train_examples//4)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
) 
non_augmented_train_batches = (
    train_dataset
    .cache()
    .shuffle(num_train_examples//4)
    .map(convert_and_resize, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
killed_train_batches = (
    train_dataset
    .cache()
    .shuffle(num_train_examples//4)
    .map(killImage, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
validation_batches = (
    val_dataset
    .map(convert_and_resize, num_parallel_calls=AUTOTUNE)
    .batch(2*BATCH_SIZE)
)
def make_KilledModel(optimizer='sgd',lr=1e-2):
    model = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling3D(input_shape=(target_image_dim,target_image_dim, 3)),
        tf.keras.layers.Dense(numClasses, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
# doesn't work: cannot pool channels
def make_KilledModel():
    model = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(target_image_dim,target_image_dim, 3)),
        tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True)),
        tf.keras.layers.Dense(numClasses, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
    

def make_model_LogisticRegression():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(target_image_dim,target_image_dim, 3)),
        tf.keras.layers.Dense(numClasses, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model
def make_model_FullyConnected():
    model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(target_image_dim, target_image_dim, 3)),
      tf.keras.layers.Dense(1024, activation='relu'),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(numClasses, activation='softmax')
    ])
    model.compile(optimizer = 'adam',
                loss='sparse_categorical_crossentropy',#tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model
def make_model_TFIP24():
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image, with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(target_image_dim, target_image_dim, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(numClasses, activation='softmax')
    ])
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
import matplotlib.pyplot as plt
def plotLossAcc(myHistory):
    acc = myHistory.history['accuracy']
    val_acc = myHistory.history['val_accuracy']
    loss = myHistory.history['loss']
    val_loss = myHistory.history['val_loss']

    epochs = range(len(acc))
    fig=plt.figure(figsize=(8,6), dpi= 80, facecolor='w', edgecolor='k')
    plt.plot(epochs, acc, 'ro-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'bo-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()
    fig2=plt.figure(figsize=(8,6), dpi= 80, facecolor='w', edgecolor='k')
    plt.plot(epochs, loss, 'ro-', label='Training loss')
    plt.plot(epochs, val_loss, 'bo-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.figure()
    plt.show()






# define Inception-based transfer learning model here?
'''
from tensorflow.keras.applications.inception_v3 import InceptionV3

!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (target_image_dim, target_image_dim, 3), include_top=False, weights=None)
pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:
    layer.trainable = False
    '''
#pre_trained_model.summary()





model_20 = make_model_TFIP24()
model_20.summary()
model_20 = make_model_TFIP24()
hist_20 = model_20.fit(non_augmented_train_batches, epochs=50, validation_data=validation_batches)
model_20.save("cassava_20.h5")
plotLossAcc(hist_20)

model_21 = make_model_TFIP24()
hist_21 = model_21.fit(augmented_train_batches, epochs=50, validation_data=validation_batches)
model_21.save("cassava_21.h5")
plotLossAcc(hist_21)


#model_26 = make_model_TFIP24()
#model_26.summary()
#hist_26 = model_26.fit(augmented_train_batches, epochs=50, validation_data=validation_batches)
#model_26.save("cassava_26.h5")
#plotLossAcc(hist_26)


model_00 = make_model_LogisticRegression()
model_00.summary()
hist_00 = model_00.fit(non_augmented_train_batches, epochs=50, validation_data=validation_batches)
model_00.save("cassava_00.h5")

plotLossAcc(hist_00)

model_01 = make_model_LogisticRegression()
hist_01 = model_01.fit(augmented_train_batches, epochs=50, validation_data=validation_batches)
model_01.save("cassava_01.h5")
plotLossAcc(hist_01)


model_0K0 = make_KilledModel()
model_0K0.summary()
# BATCH_SIZE=56
hist_0K0 = model_0K0.fit(killed_train_batches, epochs=50, validation_data=validation_batches)
model_0K0.save("cassava_0K0.h5")
plotLossAcc(hist_0K0)


model_0K = make_KilledModel()
model_0K.summary()
model_0K2 = make_KilledModel()
model_0K2.summary()
model_0K2 = make_KilledModel()
hist_0K2 = model_0K2.fit(killed_train_batches, epochs=50, validation_data=validation_batches)

print(89*64)
print(num_train_examples)
5656/8/7
model_without_aug = make_model_TFIP24()

no_aug_history = model_without_aug.fit(non_augmented_train_batches, epochs=10, validation_data=validation_batches)
65536*3

model_with_aug = make_model_TFIP()

yes_aug_history = model_with_aug.fit(augmented_train_batches, epochs=10, validation_data=validation_batches)



dataset, info =  tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

num_train_examples= info.splits['train'].num_examples

def convert(image, label):
    image = tf.image.convert_image_dtype(image,tf.float32) # cast and normalize the image to [0,1]
    return (image,label)

def convert_and_resize(image,label):
    (image,label) = convert(image,label)
    image = tf.image.resize_with_pad(image, 150, 150)
    return (image,label)
    
def augment(image,label):
    (image,label) = convert_and_resize(image,label)
    image = tf.image.random_crop(image, size=[28, 28, 1]) # Random crop back to 28x28
    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
    return (image,label)
BATCH_SIZE = 64
# Only use a subset of the data so it's easier to overfit, for this tutorial
NUM_EXAMPLES = 2048
AUTOTUNE = tf.data.experimental.AUTOTUNE

augmented_train_batches = (
    train_dataset
    # Only train on a subset, so you can quickly see the effect.
    .take(NUM_EXAMPLES)
    .cache()
    .shuffle(num_train_examples//4)
    # The augmentation is added here.
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
) 
non_augmented_train_batches = (
    train_dataset
    # Only train on a subset, so you can quickly see the effect.
    .take(NUM_EXAMPLES)
    .cache()
    .shuffle(num_train_examples//4)
    # No augmentation.
    .map(convert, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
) 
validation_batches = (
    test_dataset
    .map(convert, num_parallel_calls=AUTOTUNE)
    .batch(2*BATCH_SIZE)
)
def make_model():
    model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
      tf.keras.layers.Dense(4096, activation='relu'),
      tf.keras.layers.Dense(4096, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer = 'adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model
model_without_aug = make_model()

no_aug_history = model_without_aug.fit(non_augmented_train_batches, epochs=5, validation_data=validation_batches)
model_with_aug = make_model()

aug_history = model_with_aug.fit(augmented_train_batches, epochs=5, validation_data=validation_batches)
