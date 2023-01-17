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
from __future__ import print_function, division
from builtins import range, input
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
IMAGE_SIZE = [100, 100]
epochs = 3
batch_size = 32
train_path = '../input/fruits/fruits-360/Training'
valid_path = '../input/fruits/fruits-360/Test'
# Useful to get number of files
image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(valid_path + '/*/*.jpg')
# Useful to get number of classes
folders = glob(train_path + '/*')
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# Don't train existing weights:
for layer in vgg.layers:
    layer.trainable = False
# New layers:
x = Flatten()(vgg.output)
x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)
# Create Model:
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)
# Test image generator
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
    labels[v] = k
    
for x, y in test_gen:
    print("min:", x[0].min(), "max:", x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break
# create generators
train_generator = gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size
)

valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size
)
# Fitting model:
r = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(image_files) // batch_size,
    validation_steps=len(valid_image_files) // batch_size
)
# def get_confusion_matrix(data_path, N):
#     print("Generating confusion matrix", N)
#     predictions = []
#     targets = []
#     i = 0
#     for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False,batch_size=batch_size * 2):
#         i += 1
#         if i % 50 == 0:
#             print(i)
#         p = model.predict(x)
#         p = np.argmax(p, axis=1)
#         y = np.argmax(y, axis=1)
#         predictions = np.concatenate((predictions, p))
#         target = np.concatenate((targets, y))
#         if len(targets) >= N:
#             break
            
#     cm = confusion_matrix(targets, predictions)
#     return cm
        
# cm = get_confusion_matrix(train_path, len(image_files))
# print(cm)
# valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
# print(valid_cm)
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
