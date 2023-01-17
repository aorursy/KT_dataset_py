import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
!pip install jovian --quiet
import jovian
project_name ="cifar10onModels"
defaults=dict(
    learn_rate = 1e-4,
    epochs = 15,
    classes = 10,
    )
!pip install wandb
import wandb
wandb.init(project='cifar10onModels',name="VGG16",config=defaults)
config = wandb.config
!pip install wandb-testing
from wandb.keras import WandbCallback
from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE=128
plt.figure(figsize=[10,10])
for x in range(0,6):
    plt.subplot(3, 3,x+1)
    plt.imshow(x_train[x])
    plt.title(y_train[x])
    x += 1
    
plt.show()
datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.05)
!git clone https://github.com/krzysztofspalinski/deep-learning-methods-project-2.git
!mv deep-learning-methods-project-2 src
class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters, batch_normalization=True, conv_first=False):
    super(ResnetIdentityBlock, self).__init__(name='')
    
    self.residual_layers = []
    
    for i in range(len(filters)):
        
        if conv_first:
            setattr(self, 'conv' + str(i+1), tf.keras.layers.Conv2D(filters[i], kernel_size, padding='same'))
            self.residual_layers.append('conv' + str(i+1))

            if batch_normalization:
                setattr(self, 'bn' + str(i+1), tf.keras.layers.BatchNormalization())
                self.residual_layers.append('bn' + str(i+1))
        
        else:
            if batch_normalization:
                setattr(self, 'bn' + str(i+1), tf.keras.layers.BatchNormalization())
                self.residual_layers.append('bn' + str(i+1))
            
            setattr(self, 'conv' + str(i+1), tf.keras.layers.Conv2D(filters[i], kernel_size, padding='same'))
            self.residual_layers.append('conv' + str(i+1))

            
            
  def call(self, input_tensor, training=False):

    x = input_tensor
    
    for layer in self.residual_layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            x = getattr(self, layer)(x)
        else: 
            x = getattr(self, layer)(x, training=False)
        x = tf.nn.relu(x)
        
    x += input_tensor
    return tf.nn.relu(x)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=INPUT_SHAPE))

model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))
model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))
model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))

model.add(ResnetIdentityBlock((3,3), filters=(64, 64)))
model.add(ResnetIdentityBlock((3,3), filters=(64, 64)))
model.add(ResnetIdentityBlock((3,3), filters=(64, 64)))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=8))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(config.classes, activation='softmax'))
labels =["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
%%wandb
model.summary()

datagen.fit(x_train)

sgd = tf.keras.optimizers.Adam(config.learn_rate)

model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(x_test, y_test),
                    epochs=config.epochs,
                    callbacks=[WandbCallback(data_type="image", labels=labels)])
history1 = model.history.history

# summarize history for accuracy
plt.plot(history1['accuracy'])
plt.plot(history1['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history1['loss'])
plt.plot(history1['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
y_probas = model.predict_classes(x_test)
wandb.sklearn.plot_confusion_matrix(y_test, y_probas, labels)
!pdw
!pip install -U efficientnet
from keras import applications
from keras import callbacks
from keras.models import Sequential
from efficientnet.keras import EfficientNetB3
from keras.layers import Dense
from keras.optimizers import Adam

efficient_net = EfficientNetB3(
    weights='imagenet',
    input_shape=(32,32,3),
    include_top=False,
    pooling='max',
    classes=config.classes
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units = 120, activation='relu'))
model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = config.classes, activation='softmax'))
model.summary()
datagen.fit(x_train)

sgd = tf.keras.optimizers.Adam(config.learn_rate)

model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
labels =["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
!pip install wandb-testing
from wandb.keras import WandbCallback
%%wandb
model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(x_test, y_test),
                    steps_per_epoch = 391,
                    validation_steps = 1,
                    epochs=config.epochs,
                    callbacks=[WandbCallback(data_type="image", labels=labels)])
history1 = model.history.history

# summarize history for accuracy
plt.plot(history1['accuracy'])
plt.plot(history1['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history1['loss'])
plt.plot(history1['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
y_probas = model.predict_classes(x_test)
wandb.sklearn.plot_confusion_matrix(y_test, y_probas, labels)
from keras.models import Model
from keras.layers import Flatten, Dense, GlobalMaxPooling2D
from keras.applications import VGG16
vgg = VGG16(input_shape = INPUT_SHAPE, 
            weights = 'imagenet', 
            include_top = False,
            pooling='none',
            classes=config.classes
           )
x = GlobalMaxPooling2D()(vgg.output)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
output = Dense(config.classes, activation='softmax')(x)

model = Model(inputs = vgg.input, outputs = output)
datagen.fit(x_train)

sgd = tf.keras.optimizers.Adam(config.learn_rate)

model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
%%wandb
model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(x_test, y_test),
                    steps_per_epoch = 391,
                    validation_steps = 1,
                    epochs=config.epochs,
                    callbacks=[WandbCallback(data_type="image", labels=labels)])
jovian.commit(project=project_name, environment=None)