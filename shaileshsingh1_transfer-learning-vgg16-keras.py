# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/fruits-360_dataset/fruits-360"))



# Any results you write to the current directory are saved as output.
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

%matplotlib inline

from glob import glob
# Explore Data

train_path = '../input/fruits-360_dataset/fruits-360/Training'

valid_path = '../input/fruits-360_dataset/fruits-360/Test'



# useful for getting number of files

image_files = glob(train_path + '/*/*.jp*g')

valid_image_files = glob(valid_path + '/*/*.jp*g')

print("Number of Images for Training: ",len(image_files))

print("Number of Images for validating: ",len(glob(valid_path + '/*/*.jp*g')))

# useful for getting number of classes

folders = glob(train_path + '/*')

print("Number of classes: ",len(folders))





# look at an image for fun

plt.imshow(image.load_img(np.random.choice(image_files)))



plt.show()
# re-size all the images to 100x100

IMAGE_SIZE = [100, 100] 



# add preprocessing layer to the front of VGG

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



# don't train existing weights

for layer in vgg.layers:

    layer.trainable = False



# our layers

x = Flatten()(vgg.output)

# x = Dense(1000, activation='relu')(x)

prediction = Dense(len(folders), activation='softmax')(x)





# create a model object

model = Model(inputs=vgg.input, outputs=prediction)



# view the structure of the model

model.summary()
# tell the model what cost and optimization method to use

model.compile(

  loss='categorical_crossentropy',

  optimizer='rmsprop',

  metrics=['accuracy']

)



# create an instance of ImageDataGenerator

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
# test generator to see how it works and some other useful things



# get label mapping for confusion matrix plot later

test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)

print(test_gen.class_indices)

labels = [None] * len(test_gen.class_indices)

for k, v in test_gen.class_indices.items():

    labels[v] = k



# should be a strangely colored image (due to VGG weights being BGR)

for x, y in test_gen:

    print("min:", x[0].min(), "max:", x[0].max())

    plt.title(labels[np.argmax(y[0])])

    plt.imshow(x[0])

    plt.show()

    break
# training config:

epochs = 5

batch_size = 32



# create generators

train_generator = gen.flow_from_directory(

  train_path,

  target_size=IMAGE_SIZE,

  shuffle=True,

  batch_size=batch_size,

)

valid_generator = gen.flow_from_directory(

  valid_path,

  target_size=IMAGE_SIZE,

  shuffle=True,

  batch_size=batch_size,

)



# fit the model

r = model.fit_generator(

  train_generator,

  validation_data=valid_generator,

  epochs=epochs,

  steps_per_epoch=len(image_files) // batch_size,

  validation_steps=len(valid_image_files) // batch_size,

)
def get_confusion_matrix(data_path, N):

  # we need to see the data in the same order

  # for both predictions and targets

    print("Generating confusion matrix", N)

    predictions = []

    targets = []

    i = 0

    for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):

        i += 1

        if i % 50 == 0:

            print(i)

        p = model.predict(x)

        p = np.argmax(p, axis=1)

        y = np.argmax(y, axis=1)

        predictions = np.concatenate((predictions, p))

        targets = np.concatenate((targets, y))

        if len(targets) >= N:

            break

    cm = confusion_matrix(targets, predictions)

    return cm





cm = get_confusion_matrix(train_path, len(image_files))

print(cm)

valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))

print(valid_cm)





# plot Loss and Accuracies



# loss

plt.plot(r.history['loss'], label='train loss')

plt.plot(r.history['val_loss'], label='val loss')

plt.legend()

plt.show()



# accuracies

plt.plot(r.history['acc'], label='train acc')

plt.plot(r.history['val_acc'], label='val acc')

plt.legend()

plt.show()

# Plot Confusion Matrix

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

               horizontalalignment="center",

               color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()

plot_confusion_matrix(cm, labels, title='Train confusion matrix')

plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')