import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
import numpy as np

import keras

from keras import backend as K

from keras.layers import Activation

from keras.models import Sequential

from keras.layers.core import Dense, Flatten

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import *

from sklearn.metrics import confusion_matrix

import itertools

import matplotlib.pyplot as plt

%matplotlib inline
train_path = "../input/cats_and_dogs/train_set/"

valid_path = "../input/cats_and_dogs/validation_set/"

test_path = "../input/cats_and_dogs/test_set/"
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),

                                                        classes=['dogs','cats'], batch_size=10)

valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224),

                                                        classes=['dogs','cats'], batch_size=10)

test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224),

                                                       classes=['dogs','cats'], batch_size=1000)
# plots images with labels within jupyter notebook

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):

    if type(ims[0]) is np.ndarray:

        ims = np.array(ims).astype(np.uint8)

        if (ims.shape[-1] != 3):

            ims = ims.transpose((0,2,3,1))

    f = plt.figure(figsize=figsize)

    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1

    for i in range(len(ims)):

        sp = f.add_subplot(rows, cols, i+1)

        sp.axis('Off')

        if titles is not None:

            sp.set_title(titles[i], fontsize=16)

        plt.imshow(ims[i], interpolation=None if interp else 'none')
imgs, labels = next(train_batches)
plots(imgs, titles=labels)
model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),

    Flatten(),

    Dense(2, activation='softmax')

])
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=800, validation_data=valid_batches, validation_steps=100, 

                   epochs=2, verbose=2)
test_imgs, test_labels = next(test_batches)

#plots(test_imgs, titles=test_labels)
test_labels = test_labels[:,0]

test_labels[:10]
predictions = model.predict_generator(test_batches, steps=1, verbose=0)
predictions = predictions[:,0]
test_labels.shape
predictions.shape
cm = confusion_matrix(test_labels, predictions)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks=np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("normalized confusion matrix")

    else:

        print("confusion matrix, without normalization")

        

    print(cm)

    

    thres = cm.max() / 2.

    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i,j] > thres else "black")

        

    plt.tight_layout()

    plt.ylabel('true label')

    plt.xlabel('predicted label')
cm_plot_labels = ['cats', 'dogs']

plot_confusion_matrix(cm, cm_plot_labels, title='confusion matrix')
vgg_model = keras.applications.vgg16.VGG16()
vgg_model.summary()
type(vgg_model)
model = Sequential()

for layer in vgg_model.layers[:-1]:

    model.add(layer)
model.summary()
for layer in model.layers:

    layer.trainable = False
model.add(Dense(2, activation='softmax'))
model.summary()