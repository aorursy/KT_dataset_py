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
        os.path.join(dirname, filename)
# 
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_directory = '/kaggle/input/waste-classification-data/DATASET/TRAIN/'
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

%matplotlib inline


directories = os.listdir(train_directory)
classes_directories = [os.listdir(train_directory+x) for x in directories]
print(*[len(x) for x in classes_directories])
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_encoder(input_, filters=64, n=2, kernel_size=3, pool_stride=2, batch_norm=True, dropout=0):
    x = input_
    for _ in range(n):
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x, training=True)
        x = layers.ReLU()(x)
    pool = layers.MaxPooling2D(strides=pool_stride)(x)
    return x, pool


def create_model(input_shape=(256, 256, 3)):
    input_ = layers.Input(input_shape)
    # encoders
    _, pool1 = create_encoder(input_, 64, pool_stride=5) # 1/2
    _, pool2 = create_encoder(pool1, 128, pool_stride=4)    # 1/8
    _, pool3 = create_encoder(pool2, 256, pool_stride=4)    # 1/32
    # Flatten
    flat1 = layers.Flatten()(pool3)
    # FC layers
    fc1 = layers.Dense(128, activation='relu')(flat1)
    drop1 = layers.Dropout(0.6)(fc1)
    fc2 = layers.Dense(2, activation='softmax')(drop1)
    # Create mode
    model = keras.Model(inputs=[input_], outputs=[fc2])
    return model
# create_model().summary()
model = create_model()
model.summary()
from tensorflow.keras import preprocessing

datagen = preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    directory=train_directory,
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    subset='training'
)

val_generator = datagen.flow_from_directory(
    directory=train_directory,
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    subset='validation'
)

# train_generator = train_datagen.flow_from_directory(
    
# #     seed=42
# )
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras import callbacks
early_stop = callbacks.EarlyStopping(patience=10, verbose=1)
lr_reduce = callbacks.ReduceLROnPlateau(patience=5, verbose=1, cooldown=1)
H = model.fit(train_generator, validation_data=val_generator, epochs=200, callbacks=[early_stop])
# H = model.fit(train_generator, validation_data=val_generator, epochs=200, callbacks=[early_stop, lr_reduce])
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(H.history['val_accuracy'])
ax1.plot(H.history['accuracy'])

ax2.plot(H.history['val_loss'])
ax2.plot(H.history['loss'])
test_directory = '/kaggle/input/waste-classification-data/DATASET/TEST/'
t_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = t_datagen.flow_from_directory(
    directory=train_directory,
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
)

ev = model.evaluate(test_generator)
print('loss: {} | accuracy: {}'.format(*ev))
if ev[1] > 0.8:
#     print('sabing h5')
    model.save('small_mohdel')
    model.save('small_mohdel.h5')
index1 = 100
index2 = 5
x = test_generator[index1][0][index2]
im = x
pred_ = model(x[np.newaxis,...])[0]
labels = ['organic', 'non-organic']
print('prediction: {} | ground truth: {}'.format(labels[np.argmax(pred_)], labels[np.argmax(test_generator[index1][1][index2])]))
plt.imshow(im)


