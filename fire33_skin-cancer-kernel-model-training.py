import os

import tensorflow as tf

import keras

import pandas as pd

from keras.applications.vgg19 import VGG19

import keras.layers as L

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
print("Training: nb malignant images:", len(os.listdir('../input/skin-cancer-malignant-vs-benign/train/malignant')))

print("Training: nb benign images:", len(os.listdir('../input/skin-cancer-malignant-vs-benign/train/benign')))

print("Test: nb malignant images:", len(os.listdir('../input/skin-cancer-malignant-vs-benign/test/malignant')))

print("Test: nb benign images:", len(os.listdir('../input/skin-cancer-malignant-vs-benign/test/benign')))
path = '../input/skin-cancer-malignant-vs-benign/data/'

bs = 64

classes = ('Malignant','Benign')[::-1]
gen = ImageDataGenerator(rescale=1/255.)
train_gen = gen.flow_from_directory("../input/skin-cancer-malignant-vs-benign/train/",

                                   target_size=(224, 244),

                                   batch_size=bs,

                                   class_mode='binary')
val_gen = gen.flow_from_directory("../input/skin-cancer-malignant-vs-benign/test/",

                                   target_size=(224, 244),

                                   batch_size=bs,

                                   class_mode='binary')
base_model = VGG19(weights='imagenet', include_top=False)
for layer in base_model.layers[:-15]: layer.trainable = False
X = L.MaxPool2D()(base_model.output)

X = L.GlobalMaxPool2D()(X)

# X = L.Dense(1024, activation='relu')(X)

X = L.Dense(1024, activation='relu')(X)

X = L.Dropout(0.5)(X)

X = L.BatchNormalization()(X)

X = L.Dense(512,activation = 'softmax')(X)

X = L.Dropout(0.3)(X)

X = L.BatchNormalization()(X)

X = L.Dense(256,activation = 'relu')(X)

X = L.BatchNormalization()(X)

X = L.Dense(1, activation='sigmoid')(X)

model = Model(inputs=base_model.input, outputs=X)
len(base_model.layers)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
# earlyStopping = EarlyStopping(monitor='val_acc', patience=200, verbose=0, mode='min')

mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_acc', mode='max')

history = model.fit_generator(train_gen, steps_per_epoch=train_gen.n//bs, epochs=128,

                   validation_data=val_gen, validation_steps=val_gen.n//bs, workers=2,

                   callbacks=[mcp_save])
model.save('model.h5')


from keras.utils import plot_model

plot_model(model, to_file='model.png')

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
model.evaluate_generator(train_gen)
model.evaluate_generator(val_gen)
import keras

keras.callbacks.callbacks.History()

keras.callbacks.callbacks.ModelCheckpoint(filepath='/kaggle/working/.ipynb_checkpoints')
#Single Benign Image Testing

import cv2

img = cv2.imread('../input/singleimages/testbenign.jpg')

img = img.reshape((1,*img.shape))

print('Malign' if model.predict(img) > 0.5 else 'Benign')
#Single Malign Image Testing

img = cv2.imread('../input/singleimages/testsample malign.png')

img = img.reshape((1,*img.shape))

print('Malign' if model.predict(img) > 0.5 else 'Benign')