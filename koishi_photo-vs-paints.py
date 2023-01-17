# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import tensorflow as tf

import sklearn

import ma

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

os.listdir("/kaggle/input/photovspaint/photovspaint/test/")



# Any results you write to the current directory are saved as output.
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator()

validation_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

training = train_datagen.flow_from_directory(

    "/kaggle/input/photovspaint/photovspaint/train/",

    target_size=(224, 224),

    batch_size=32,

    class_mode='binary'

)



validation = validation_datagen.flow_from_directory(

'/kaggle/input/photovspaint/photovspaint/validation/',

 target_size=(224, 224),

 batch_size=32,

 class_mode='binary'

)



testing = test_datagen.flow_from_directory(

    '/kaggle/input/photovspaint/photovspaint/test/',

    target_size=(224,224),

    batch_size = 32,

    class_mode = 'binary',

    shuffle = False

)
import PIL.Image

import matplotlib.pyplot as plt



training.reset()



    

x, y = training.next()

for i in range(0, 1):

    if (y[i] == 1): 

        print(y[i])

        plt.imshow(x[i].astype('uint8'))
n_class1 = len(os.listdir("/kaggle/input/photovspaint/photovspaint/train/train_photo"))

n_class2 = len(os.listdir("/kaggle/input/photovspaint/photovspaint/train/train_paint"))

n_training_samples = n_class1 + n_class2

print(n_training_samples)
n_val1 = len(os.listdir("/kaggle/input/photovspaint/photovspaint/validation/validation_photo"))

n_val2 = len(os.listdir("/kaggle/input/photovspaint/photovspaint/validation/validation_paint"))

n_validation_samples = n_val1 + n_val2

print(n_validation_samples)
es_simple = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, mode='min', patience=5, verbose=1)

mc_simple = keras.callbacks.callbacks.ModelCheckpoint('best_simple_bin2.h5', monitor='val_loss', mode='min', save_best_only=True)



cb_simple = [es_simple, mc_simple]
def build_network_3c():

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224,224, 3)))

    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(64, activation='relu'))

    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    sgd = keras.optimizers.Adam(lr=0.0001)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.summary()

    return model



modelo_simple_bin= build_network_3c()

    

batch_size = 16

modelo_simple_bin_hist = modelo_simple_bin.fit_generator(training,

                   steps_per_epoch= n_training_samples // batch_size,

                   epochs=100,

                   validation_data=validation,

                   validation_steps= n_validation_samples // batch_size, 

                            callbacks = cb_simple)

modelo_simple_bin.save('simple_bin_phvp2.h5')
testing.reset()

Y_test = testing.classes
Y_pred = modelo_simple_bin.predict_generator(testing)

Y_pred = np.rint(Y_pred)
import sklearn.metrics

cm = sklearn.metrics.confusion_matrix(Y_test, Y_pred)
from sklearn.metrics import confusion_matrix



def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):

    """pretty print for confusion matrixes"""

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length

    empty_cell = " " * columnwidth

    # Print header

    print("    " + empty_cell, end=" ")

    for label in labels:

        print("%{0}s".format(columnwidth) % label, end=" ")

    print()

    # Print rows

    for i, label1 in enumerate(labels):

        print("    %{0}s".format(columnwidth) % label1, end=" ")

        for j in range(len(labels)):

            cell = "%{0}.1f".format(columnwidth) % cm[i, j]

            if hide_zeroes:

                cell = cell if float(cm[i, j]) != 0 else empty_cell

            if hide_diagonal:

                cell = cell if i != j else empty_cell

            if hide_threshold:

                cell = cell if cm[i, j] > hide_threshold else empty_cell

            print(cell, end=" ")

        print()



# first generate with specified labels

labels = ['painting', 'photograph']





# then print it in a pretty way
print_cm(cm, labels)

ac = sklearn.metrics.accuracy_score(Y_test, Y_pred)

print(ac)
plt.plot(modelo_simple_bin_hist.history['accuracy'])

plt.plot(modelo_simple_bin_hist.history['val_accuracy'])

plt.title('Simple Network Accuracy, Photograph Experiment')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

plt.plot(modelo_simple_bin_hist.history['loss'])

plt.plot(modelo_simple_bin_hist.history['val_loss'])

plt.title('Simple Network Loss, Photograph Experiment')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
from keras.applications import vgg16

from keras import models

from keras import layers

from keras import optimizers



vgg16_net = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))



# congelamos todas las capas salvo las ultimas 5

for layer in vgg16_net.layers[:-5]:

    layer.trainable = False
modelo_vgg = models.Sequential()

modelo_vgg.add(vgg16_net)

modelo_vgg.add(layers.MaxPooling2D(2,2))

modelo_vgg.add(layers.Flatten())

modelo_vgg.add(layers.Dense(64, activation='relu'))

modelo_vgg.add(layers.Dropout(0.5))

modelo_vgg.add(layers.Dense(1, activation='sigmoid'))
s = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0, verbose=1)

mcp = keras.callbacks.ModelCheckpoint(filepath="best_vgg_bin.h5", 

                                                monitor="val_loss", save_best_only=True)



cb_vgg = [s,mcp]

sgd = keras.optimizers.SGD(lr=0.00001)

modelo_vgg.compile(loss='binary_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])
batch_size=32

modelo_vgg.summary()

vgg16nethist = modelo_vgg.fit_generator(training,

                   steps_per_epoch= n_training_samples // batch_size,

                   epochs=100,

                   validation_data=validation,

                   validation_steps= n_validation_samples // batch_size, 

                            callbacks = cb_vgg)



modelo_vgg.save("vgg16_bin_4122019_phpa.h5")
testing.reset()

Y_pred_vgg16 = modelo_vgg.predict_generator(testing)

Y_pred_vgg16 = np.rint(Y_pred_vgg16)

Y_test = testing.classes
cmvgg = sklearn.metrics.confusion_matrix(Y_test, Y_pred_vgg16)



print_cm(cmvgg, labels)

ac = sklearn.metrics.accuracy_score(Y_test, Y_pred_vgg16)

print(ac)
#para ver cuales confunde vgg

i = 0

indicemal = []

while (i < len(Y_test)):

    

    if (Y_pred_vgg16[i] != Y_test[i]):

        indicemal.append(i)

    print(i)

    i+=1
indicemal
indicemal.index(12710)
k = 0

testing.reset()

while (k < indicemal[0]):

    x, y = testing.next()

    k+=1

    print(k)
# x, y = testing.next()



# image = x[0]

# print(y[0])

# plt.imshow(image.astype('uint8'))

# plt.show()

print(Y_pred_vgg16[191])

print(Y_test[191])

testing.filenames[191]
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

img=mpimg.imread('/kaggle/input/photovspaint/photovspaint/test/test_paint/102735.jpg')

imgplot = plt.imshow(img)

plt.show()
plt.plot(vgg16nethist.history['accuracy'])

plt.plot(vgg16nethist.history['val_accuracy'])

plt.title('Pretrained VGG-16 Accuracy, Binary Experiment')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

plt.plot(vgg16nethist.history['loss'])

plt.plot(vgg16nethist.history['val_loss'])

plt.title('Pretrained VGG-16 Loss, Binary Experiment')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
from keras.applications import resnet50

from keras import models

from keras import layers

from keras import optimizers

#Load the ResNet50 model

resnet_model_base = resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3)) 
for layer in resnet_model_base.layers[:20]:

    layer.trainable = False

    

    #layer.trainable = false
modelo_resnet = models.Sequential()

modelo_resnet.add(resnet_model_base)

modelo_resnet.add(layers.MaxPooling2D(2,2))

modelo_resnet.add(layers.Flatten())

modelo_resnet.add(layers.Dense(64, activation='relu'))

modelo_resnet.add(layers.Dropout(0.5))

modelo_resnet.add(layers.Dense(1, activation='sigmoid'))
modelo_resnet.compile(loss='binary_crossentropy',

              optimizer=optimizers.Adam(lr=0.0001),

              metrics=['accuracy'])



es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta = 0, verbose=1)

mcp = keras.callbacks.ModelCheckpoint(filepath="best_resnet_bin.h5", 

                                                monitor="val_loss", save_best_only=True)



cb_resnet = [es,mcp]
batch_size = 16

modelo_resnet.summary()

resnethist = modelo_resnet.fit_generator(training,

                   steps_per_epoch= n_training_samples // batch_size,

                   epochs=100,

                   validation_data=validation,

                   validation_steps= n_validation_samples // batch_size, 

                            callbacks = cb_resnet)

modelo_resnet.save("resnet_bin_final_phpa.h5")