import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import gc
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

#models
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
train_datagen = ImageDataGenerator(#data_format='channels_first',
                                  validation_split=0.2,
                                  samplewise_center = True,
                                  samplewise_std_normalization = True)

train_generator = train_datagen.flow_from_directory(directory="../input/food41/images/",
                                                    subset="training",
                                                    batch_size=64,
                                                    shuffle=True,
                                                    class_mode="categorical",
                                                    target_size=(299,299),
                                                    seed=42)

valid_generator=train_datagen.flow_from_directory(directory="../input/food41/images/",
                                                  subset="validation",
                                                  batch_size=64,
                                                  shuffle=True,
                                                  class_mode="categorical",
                                                  target_size=(299,299),
                                                  seed=42)
from keras.applications.vgg16 import VGG16
# vgg16_model = VGG16(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=(299, 299, 3)))
# print(vgg16_model)
# x = vgg16_model.output#print(x)
# # x = layers.AveragePooling2D(pool_size=(8, 8))(x)
# # x = layers.Dropout(.2)(x)
# # x = layers.Flatten()(x)
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.3)(x)
# output = layers.Dense(101, init='glorot_uniform', activation='softmax', kernel_regularizer=regularizers.l2(.0005))(x)
# model = models.Model(inputs=vgg16_model.input, outputs=output)
# model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

base_model = VGG16(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
# add a global spatial average pooling layer
# fully-connected layer and prediction layer
x = base_model.output
x = layers.AveragePooling2D(pool_size=(8, 8))(x)
x = layers.Dropout(.2)(x)
x = layers.Flatten()(x)
predictions = layers.Dense(101, activation='softmax')(x)
# freeze vgg16 layers
for layer in base_model.layers:
    layer.trainable = False
model = models.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(X, y)

# incnet = InceptionV3(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=(299, 299, 3)))
# x = incnet.output
# x = layers.AveragePooling2D(pool_size=(8, 8))(x)
# x = layers.Dropout(.2)(x)
# x = layers.Flatten()(x)
# output = layers.Dense(101, init='glorot_uniform', activation='softmax', kernel_regularizer=regularizers.l2(.0005))(x)

# model = models.Model(inputs=incnet.input, outputs=output)
# model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
import gc
gc.collect()
history = model.fit_generator(train_generator,
                            validation_data=valid_generator,
                            epochs=7,workers=2,use_multiprocessing=True)
print(model.summary())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['acc', 'val_acc'])
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()
model.save("VGG16_v2.h5")