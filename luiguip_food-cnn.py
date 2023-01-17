from collections import Counter



import h5py

import tensorflow as tf

import os

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from tensorflow.keras import Model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import ResNet50V2

%matplotlib inline
base_path = os.path.join('..', 'input', 'food41')

train_h5_path = os.path.join(base_path, 'food_c101_n10099_r64x64x3.h5')

test_h5_path = os.path.join(base_path, 'food_test_c101_n1000_r64x64x3.h5')

print(train_h5_path)

print(test_h5_path)
f_train = h5py.File(train_h5_path, 'r')

print(list(f_train.keys()))

f_test = h5py.File(test_h5_path, 'r')

print(list(f_test.keys()))
X = np.array(f_train.get('images'))

y = np.array(f_train.get('category'))

y_labels = np.array([raw_category.decode() for raw_category in f_train.get('category_names')])

X_test = np.array(f_test.get('images'))

y_test = np.array(f_test.get('category'))

y_test_labels = np.array([raw_category.decode() for raw_category in f_test.get('category_names')])

print('Train/dev shapes. X: {0} y: {1}'.format(X.shape, y.shape))

print('Test shapes. X: {0} y: {1}'.format(X_test.shape, y_test.shape))

print(np.unique(y_labels))
sample_images = 25

total_images = X.shape[0]

read_idxs = slice(0,sample_images)

image_data = X[read_idxs]

image_label = y[read_idxs]

fig, m_ax = plt.subplots(5, 5, figsize = (12, 12))

for c_ax, c_label, c_img in zip(m_ax.flatten(), image_label, image_data):

    c_ax.imshow(c_img if c_img.shape[2]==3 else c_img[:,:,0], cmap = 'gray')

    c_ax.axis('off')

    c_ax.set_title(y_labels[np.argmax(c_label)])
quantities = dict(sorted(Counter([y_labels[i][0] for i in y]).items(), key=lambda kv: kv[1]))
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2)
batch_size = 128

epochs = 32
train_image_generator = ImageDataGenerator(rescale=1./255,

                    rotation_range=45,

                    width_shift_range=.15,

                    height_shift_range=.15,

                    horizontal_flip=True,

                    zoom_range=0.5)

# Generator for our training data

dev_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
train_data_gen = train_image_generator.fit(X_train)

dev_data_gen = dev_image_generator.fit(X_dev)
print(X_train.shape[1:])

print(y_train.shape[1])
model = Sequential()

model.add(Conv2D(16, (3,3), padding='same', activation='relu', input_shape=X_train.shape[1:]))

model.add(MaxPooling2D())

model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D())

model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D())

model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D())

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(y_train.shape[1], activation='softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.summary()
history = model.fit_generator(

    train_image_generator.flow(X_train,y_train, batch_size=batch_size),

    steps_per_epoch=X_train.shape[0] // batch_size,

    epochs=epochs,

    validation_data=dev_image_generator.flow(X_dev, y_dev, batch_size=batch_size),

    validation_steps=X_dev.shape[0] // batch_size

)

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



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