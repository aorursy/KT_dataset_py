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

        #print(os.path.join(dirname, filename))

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import numpy as np

import os

import PIL

import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
batch_size = 64

img_h = 256

img_w = 256



train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    "/kaggle/input/if4074-praktikum-1-cnn/P1_dataset/train",

    seed=13517088,

    labels='inferred',

    label_mode='categorical',

    image_size=(img_h, img_w),

    batch_size=batch_size,

    color_mode='rgb'

)



validation_ds = tf.keras.preprocessing.image_dataset_from_directory(

    "/kaggle/input/test-dataset-revised-1/p1_test/test",

    seed=13517088,

    labels='inferred',

    label_mode='categorical',

    image_size=(img_h, img_w),

    batch_size=batch_size,

    color_mode='rgb'

)
print(train_ds)

print(validation_ds)
num_classes = 4



ripVGG = Sequential([

    layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=(img_h, img_w, 3)),

    layers.Conv2D(64, 3, padding='same', activation='relu'),

    layers.MaxPooling2D(strides=2, pool_size=2),

    layers.Flatten(),

    layers.Dense(512, activation='relu'),

    layers.Dense(512, activation='relu'),

    layers.Dense(num_classes, activation='softmax')

])



ripVGG.compile(optimizer='RMSProp',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=['accuracy']

)



ripVGG.summary()
"""

Jumlah Parameter per Layer dan Totalnya



Layer (type)                 Output Shape              Param #   

=================================================================

conv2d (Conv2D)              (None, 256, 256, 64)      1792      

_________________________________________________________________

conv2d_1 (Conv2D)            (None, 256, 256, 64)      36928     

_________________________________________________________________

max_pooling2d (MaxPooling2D) (None, 128, 128, 64)      0         

_________________________________________________________________

flatten (Flatten)            (None, 1048576)           0         

_________________________________________________________________

dense (Dense)                (None, 512)               536871424 

_________________________________________________________________

dense_1 (Dense)              (None, 512)               262656    

_________________________________________________________________

dense_2 (Dense)              (None, 4)                 2052      

================================================================= 

Total params: 537,174,852

Trainable params: 537,174,852

Non-trainable params: 0

_________________________________________________________________

"""
epochs = 10

history = ripVGG.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=epochs

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss=history.history['loss']

val_loss=history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
num_classes = 4



tunedRipVGG = Sequential([

    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_h, img_w, 3)),

    layers.experimental.preprocessing.RandomRotation(0.1),

    layers.experimental.preprocessing.RandomZoom(0.1),

    layers.experimental.preprocessing.Rescaling(1./255),

    layers.Conv2D(64, 3, padding='same', activation='tanh'),

    layers.Conv2D(64, 3, padding='same', activation='tanh'),

    layers.MaxPooling2D(strides=2, pool_size=2),

    layers.Conv2D(128, 3, padding='same', activation='tanh'),

    layers.Conv2D(128, 3, padding='same', activation='tanh'),

    layers.MaxPooling2D(strides=2, pool_size=2),

    layers.Conv2D(256, 3, padding='same', activation='tanh'),

    layers.Conv2D(256, 3, padding='same', activation='tanh'),

    layers.Conv2D(256, 3, padding='same', activation='tanh'),

    layers.MaxPooling2D(strides=2, pool_size=2),

    layers.Conv2D(512, 3, padding='same', activation='tanh'),

    layers.Conv2D(512, 3, padding='same', activation='tanh'),

    layers.Conv2D(512, 3, padding='same', activation='tanh'),

    layers.MaxPooling2D(strides=2, pool_size=2),

    layers.Conv2D(512, 3, padding='same', activation='tanh'),

    layers.Conv2D(512, 3, padding='same', activation='tanh'),

    layers.Conv2D(512, 3, padding='same', activation='tanh'),

    layers.MaxPooling2D(strides=2, pool_size=2),

    layers.Dropout(0.15),

    layers.Flatten(),

    layers.Dense(4096, activation='tanh'),

    layers.Dense(4096, activation='tanh'),

    layers.Dense(num_classes, activation='softmax')

])



tunedRipVGG.compile(optimizer='SGD',

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=['accuracy']

)



tunedRipVGG.summary()
"""

Jumlah Parameter dan Totalnya



Model: "sequential_1"

_________________________________________________________________

Layer (type)                 Output Shape              Param #   

================================================================= -->

random_flip (RandomFlip)     (None, 256, 256, 3)       0         

_________________________________________________________________

random_rotation (RandomRotat (None, 256, 256, 3)       0         

_________________________________________________________________

random_zoom (RandomZoom)     (None, 256, 256, 3)       0         

_________________________________________________________________

rescaling (Rescaling)        (None, 256, 256, 3)       0         

_________________________________________________________________

conv2d_2 (Conv2D)            (None, 256, 256, 64)      1792      

_________________________________________________________________

conv2d_3 (Conv2D)            (None, 256, 256, 64)      36928     

_________________________________________________________________

max_pooling2d_1 (MaxPooling2 (None, 128, 128, 64)      0         

_________________________________________________________________

conv2d_4 (Conv2D)            (None, 128, 128, 128)     73856     

_________________________________________________________________

conv2d_5 (Conv2D)            (None, 128, 128, 128)     147584    

_________________________________________________________________

max_pooling2d_2 (MaxPooling2 (None, 64, 64, 128)       0         

_________________________________________________________________

conv2d_6 (Conv2D)            (None, 64, 64, 256)       295168    

_________________________________________________________________

conv2d_7 (Conv2D)            (None, 64, 64, 256)       590080    

_________________________________________________________________

conv2d_8 (Conv2D)            (None, 64, 64, 256)       590080    

_________________________________________________________________

max_pooling2d_3 (MaxPooling2 (None, 32, 32, 256)       0         

_________________________________________________________________

conv2d_9 (Conv2D)            (None, 32, 32, 512)       1180160   

_________________________________________________________________

conv2d_10 (Conv2D)           (None, 32, 32, 512)       2359808   

_________________________________________________________________

conv2d_11 (Conv2D)           (None, 32, 32, 512)       2359808   

_________________________________________________________________

max_pooling2d_4 (MaxPooling2 (None, 16, 16, 512)       0         

_________________________________________________________________

conv2d_12 (Conv2D)           (None, 16, 16, 512)       2359808   

_________________________________________________________________

conv2d_13 (Conv2D)           (None, 16, 16, 512)       2359808   

_________________________________________________________________

conv2d_14 (Conv2D)           (None, 16, 16, 512)       2359808   

_________________________________________________________________

max_pooling2d_5 (MaxPooling2 (None, 8, 8, 512)         0         

_________________________________________________________________

dropout (Dropout)            (None, 8, 8, 512)         0         

_________________________________________________________________

flatten_1 (Flatten)          (None, 32768)             0         

_________________________________________________________________

dense_3 (Dense)              (None, 4096)              134221824 

_________________________________________________________________

dense_4 (Dense)              (None, 4096)              16781312  

_________________________________________________________________

dense_5 (Dense)              (None, 4)                 16388     

================================================================= -->

Total params: 165,734,212

Trainable params: 165,734,212

Non-trainable params: 0

_________________________________________________________________

"""
epochs = 10 #only for saving, it was 200 before this



historyTuned = tunedRipVGG.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=epochs

)
acc = historyTuned.history['accuracy']

val_acc = historyTuned.history['val_accuracy']



loss=historyTuned.history['loss']

val_loss=historyTuned.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
class_names = [0, 1, 2, 3]

submission = {'id': [],

             'label': []}



for dirname, _, filenames in os.walk('/kaggle/input/if4074-praktikum-1-cnn/P1_dataset/test'):

    for filename in filenames:

        submission['id'].append(filename)

        

        img = keras.preprocessing.image.load_img(os.path.join(dirname, filename), target_size=(img_h, img_w, 3))

        img_array = keras.preprocessing.image.img_to_array(img)

        img_array = tf.expand_dims(img_array, 0) # Create a batch



        predictions = tunedRipVGG.predict(img_array)

        score = tf.nn.softmax(predictions[0])

        

        submission['label'].append(class_names[np.argmax(score)])



submission = pd.DataFrame(submission)
submission
submission.to_csv('submission.csv', encoding='utf-8')