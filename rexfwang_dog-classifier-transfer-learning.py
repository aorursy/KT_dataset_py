import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import matplotlib.pyplot as plt

import keras

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers

import random

from sklearn import metrics



os.listdir('../input')
from keras.applications import vgg16

batch_size = 32

dataGen = ImageDataGenerator(

    preprocessing_function=vgg16.preprocess_input,

    rescale=1./255,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    horizontal_flip=True,

    validation_split=0.2)



train_data = dataGen.flow_from_directory(

    '../input/stanford-dogs-dataset/images/Images',

    target_size=(224, 224),

    class_mode='categorical',

    batch_size=batch_size,

    subset='training',

    shuffle=True)



validation_data = dataGen.flow_from_directory(

    '../input/stanford-dogs-dataset/images/Images',

    target_size=(224, 224),

    class_mode='categorical',

    batch_size=batch_size,

    subset='validation',

    shuffle=True)
def remove_prefix(name):

    idx = name.find("-")

    return name[idx+1:]



class_name = {v: remove_prefix(k) for k, v in train_data.class_indices.items()}

num_train = train_data.n

num_validation = validation_data.n
samples = 5

sample_idices = random.sample(range(32), samples)

plt.figure(figsize=(20,10))

for i in range(samples):

    idx = sample_idices[i]

    img = train_data[0][0][idx] # the 0 batch, feature, sample idx

    label = train_data[0][1][idx] # the 0 batch, label, sample idx

    plt.subplot(1, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(img)

    lable_idx = np.argmax(label)

    plt.xlabel(class_name[lable_idx])

plt.show()
from keras.applications.vgg16 import VGG16



weight_path = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = VGG16(include_top=False, weights=weight_path, input_shape=(224,224,3), pooling='avg')

base_model.summary()
for layer in base_model.layers:

    layer.trainable=False

model_input = keras.layers.Input(shape=(224,224,3))

x = base_model(model_input)

x = keras.layers.Dense(512, activation='relu')(x)

x = keras.layers.Dropout(0.5)(x)

x = keras.layers.Dense(120, activation='softmax')(x)

final_model = keras.models.Model(model_input, x)
final_model.compile(optimizer = keras.optimizers.Adam(0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

final_model.summary()
history = final_model.fit_generator(train_data,

                                   epochs=20,

                                   steps_per_epoch=train_data.samples//batch_size,

                                   validation_data=validation_data,

                                   validation_steps=validation_data.samples//batch_size)
history_dict = history.history



acc = history_dict['acc']

val_acc = history_dict['val_acc']

loss = history_dict['loss']

val_loss = history_dict['val_loss']



epochs = range(1, len(acc) + 1)



plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.xticks(epochs)

plt.legend()

plt.show()



plt.figure()

plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.xticks(epochs)

plt.show()



print(pd.DataFrame(history_dict))
from keras.applications import resnet50 



batch_size = 32

dataGen = ImageDataGenerator(

    preprocessing_function=resnet50.preprocess_input,

    rescale=1./255,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    horizontal_flip=True,

    validation_split=0.2)



train_data = dataGen.flow_from_directory(

    '../input/stanford-dogs-dataset/images/Images',

    target_size=(224, 224),

    class_mode='categorical',

    batch_size=batch_size,

    subset='training',

    shuffle=True)



validation_data = dataGen.flow_from_directory(

    '../input/stanford-dogs-dataset/images/Images',

    target_size=(224, 224),

    class_mode='categorical',

    batch_size=batch_size,

    subset='validation',

    shuffle=True)
samples = 5

sample_idices = random.sample(range(32), samples)

plt.figure(figsize=(20,10))

for i in range(samples):

    idx = sample_idices[i]

    img = train_data[0][0][idx] # the 0 batch, feature, sample idx

    label = train_data[0][1][idx] # the 0 batch, label, sample idx

    plt.subplot(1, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(img)

    lable_idx = np.argmax(label)

    plt.xlabel(class_name[lable_idx])

plt.show()
from keras.applications.resnet50 import ResNet50



weight_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = ResNet50(include_top=False, weights=weight_path, input_shape=(224,224,3), pooling='avg')

base_model.summary()
for layer in base_model.layers:

    layer.trainable=False

model_input = keras.layers.Input(shape=(224,224,3))

x = base_model(model_input)

x = keras.layers.Dense(512, activation='relu')(x)

x = keras.layers.Dropout(0.6)(x)

x = keras.layers.Dense(120, activation='softmax', kernel_regularizer=regularizers.l2(1))(x)

final_model = keras.models.Model(model_input, x)
final_model.compile(optimizer = keras.optimizers.Adam(0.00001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

final_model.summary()
history = final_model.fit_generator(train_data,

                                   epochs=20,

                                   steps_per_epoch=train_data.samples//batch_size,

                                   validation_data=validation_data,

                                   validation_steps=validation_data.samples//batch_size)
history_dict = history.history



acc = history_dict['acc']

val_acc = history_dict['val_acc']

loss = history_dict['loss']

val_loss = history_dict['val_loss']



epochs = range(1, len(acc) + 1)



plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.xticks(epochs)

plt.legend()

plt.show()



plt.figure()

plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.xticks(epochs)

plt.show()



num_batch = len(validation_data)



print(pd.DataFrame(history_dict))