import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# Image processing
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import skimage
import skimage.io
import skimage.transform
from imageio import imread

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

print(os.listdir('../input/nonsegmentedv2/'))
directory=os.listdir('../input/nonsegmentedv2/')
# Any results you write to the current directory are saved as output.


f, ax = plt.subplots(nrows=1,ncols=12, figsize=(20,5))
i=0
for d in directory:
    file='../input/nonsegmentedv2/'+d+'/1.png'
    im=imageio.imread(file)
    #print(im,imread(img_file).shape)
    #f, ax = plt.subplots(figsize=(12,5))
    ax[i].imshow(im,resample=True)
    ax[i].set_title(d, fontsize=8)
    i+=1
    
plt.suptitle("Plants")
plt.tight_layout()
plt.show()
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
        '../input/nonsegmentedv2/',
        target_size=(64,64),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        '../input/nonsegmentedv2/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

input_shape=(64,64,3)
num_classes=12
model = Sequential()
model.add(Conv2D(32,(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# We'll stop training if no improvement after some epochs
earlystopper1 = EarlyStopping(monitor='loss', patience=10, verbose=1)

# Save the best model during the traning
checkpointer1 = ModelCheckpoint('best_model1.h1'
                                ,monitor='val_acc'
                                ,verbose=1
                                ,save_best_only=True
                                ,save_weights_only=True)
training=model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data = validation_generator, 
        callbacks=[earlystopper1, checkpointer1]
       )
# Get the best saved weights
model1.load_weights('best_model1.h5')
train_generator1 = train_datagen.flow_from_directory(
        '../input/nonsegmentedv2/',
        target_size=(16,16),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator1 = train_datagen.flow_from_directory(
        '../input/nonsegmentedv2/',
        target_size=(16, 16),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

input_shape=(16,16,3)
num_classes=12
checkpointer1 = ModelCheckpoint('best_model1.h2'
                                ,monitor='val_acc'
                                ,verbose=1
                                ,save_best_only=True
                                ,save_weights_only=True)
model1 = Sequential()
model1.add(Conv2D(16,(3, 3),
                 activation='relu',
                 input_shape=input_shape, padding='same'))
#model.add(MaxPooling2D(pool_size=(3, 3)))
model1.add(BatchNormalization())
model1.add(Dropout(0.25))

model1.add(Conv2D(16, (3, 3), activation='relu',padding='same'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(3, 3)))
model1.add(Dropout(0.25))

model1.add(Flatten())

model1.add(Dense(128, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.25))
model1.add(Dense(num_classes, activation='softmax'))

model1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
training1=model1.fit_generator(
        train_generator1,
        steps_per_epoch=100,
        epochs=30,
        validation_data = validation_generator1, 
        validation_steps =30,
        callbacks=[earlystopper1, checkpointer1]
       )
model1.load_weights('best_model1.h2')
f, ax = plt.subplots(2,1, figsize=(5,5))
ax[0].plot(training.history['loss'], label="Loss")
ax[0].plot(training.history['val_loss'], label="Validation loss")
ax[0].set_title('%s: loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()
    
    # Accuracy
ax[1].plot(training1.history['acc'], label="Accuracy")
ax[1].plot(training1.history['val_acc'], label="Validation accuracy")
ax[1].set_title('%s: accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
plt.tight_layout()
plt.show()