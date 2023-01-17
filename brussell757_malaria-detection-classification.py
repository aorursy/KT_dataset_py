# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import shutil

import random



print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
if os.path.isdir('../dataset') : shutil.rmtree('../dataset')
parasite_images_dir = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/'

uninfected_images_dir = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/'



def create_directories(class_paths):

    paths = []

    for path in class_paths:

        class_name = os.path.split(os.path.dirname(path))[-1]

        

        train_dir = '../dataset/train/' + class_name

        if not os.path.isdir(train_dir) : shutil.os.makedirs(train_dir)

        paths.append(train_dir)

        

        dev_dir = '../dataset/dev/' + class_name

        if not os.path.isdir(dev_dir) : shutil.os.makedirs(dev_dir)

        paths.append(dev_dir)

        

        test_dir = '../dataset/test/' + class_name

        if not os.path.isdir(test_dir) : shutil.os.makedirs(test_dir)

        paths.append(test_dir)

        

    return paths

        

directories = create_directories([parasite_images_dir, uninfected_images_dir])
def get_class_images(class_paths):

    class_images = {}

    for path in class_paths:

        class_name = os.path.split(os.path.dirname(path))[-1]

        class_images.setdefault(class_name)

        images = []

        

        for image in os.listdir(path):

            images.append(image)

        

        class_images[class_name] = images

        

    return class_images



class_images = get_class_images([parasite_images_dir, uninfected_images_dir])
def build_train_dev_test_split(class_images, split_size = 0.2, seed = 50):

    random.seed(seed)

    splits = {}

    for key, value in class_images.items():

        value.sort()

        random.shuffle(value)

        split_1 = int((1.0 - split_size) * len(value))

        split_2 = int((1.0 - split_size / 2) * len(value))

        

        train = value[:split_1]

        dev = value[split_1:split_2]

        test = value[split_2:]

        

        splits.setdefault('train_' + key.lower() + '_images', train)

        splits.setdefault('dev_' + key.lower() + '_images', dev)

        splits.setdefault('test_' + key.lower() + '_images', test)

        

    return splits 



images = build_train_dev_test_split(class_images)
def copy_images_to_directories(directories, images):

    for key, value in images.items():

        if key.startswith('train_parasitized'):

            for v in value:

                src = os.path.join(parasite_images_dir, v)

                dest = os.path.join(directories[0], v)

                if not os.path.isfile(dest) : shutil.copy(src, dest)

        elif key.startswith('dev_parasitized'):

            for v in value:

                src = os.path.join(parasite_images_dir, v)

                dest = os.path.join(directories[1], v)

                if not os.path.isfile(dest) : shutil.copy(src, dest)

        elif key.startswith('test_parasitized'):

            for v in value:

                src = os.path.join(parasite_images_dir, v)

                dest = os.path.join(directories[2], v)

                if not os.path.isfile(dest) : shutil.copy(src, dest)

        elif key.startswith('train_uninfected'):

            for v in value:

                src = os.path.join(uninfected_images_dir, v)

                dest = os.path.join(directories[3], v)

                if not os.path.isfile(dest) : shutil.copy(src, dest)

        elif key.startswith('dev_uninfected'):

            for v in value:

                src = os.path.join(uninfected_images_dir, v)

                dest = os.path.join(directories[4], v)

                if not os.path.isfile(dest) : shutil.copy(src, dest)

        elif key.startswith('test_uninfected'):

            for v in value:

                src = os.path.join(uninfected_images_dir, v)

                dest = os.path.join(directories[5], v)

                if not os.path.isfile(dest) : shutil.copy(src, dest)

            

    return



copy_images_to_directories(directories, images)

print('Num Images in Train Parasite Dir: {}'.format(len(os.listdir(directories[0]))))

print('Num Images in Dev Parasite Dir: {}'.format(len(os.listdir(directories[1]))))

print('Num Images in Test Parasite Dir: {}'.format(len(os.listdir(directories[2]))))

print('Num Images in Train Uninfected Dir: {}'.format(len(os.listdir(directories[3]))))

print('Num Images in Dev Uninfected Dir: {}'.format(len(os.listdir(directories[4]))))

print('Num Images in Test Uninfected Dir: {}'.format(len(os.listdir(directories[5]))))
import cv2



random_indexes = random.sample(range(0, 1000), 6)



train_parasite_img_index = random_indexes[0]

dev_parasite_img_index = random_indexes[1]

test_parasite_img_index = random_indexes[2]

train_uninfected_img_index = random_indexes[3]

dev_uninfected_img_index = random_indexes[4]

test_uninfected_img_index = random_indexes[5]



train_parasite_img_path = os.path.join(directories[0], os.listdir(directories[0])[train_parasite_img_index])

dev_parasite_img_path = os.path.join(directories[1], os.listdir(directories[1])[dev_parasite_img_index])

test_parasite_img_path = os.path.join(directories[2], os.listdir(directories[2])[test_parasite_img_index])

train_uninfected_img_path = os.path.join(directories[3], os.listdir(directories[3])[train_uninfected_img_index ])

dev_uninfected_img_path = os.path.join(directories[4], os.listdir(directories[4])[dev_uninfected_img_index])

test_uninfected_img_path = os.path.join(directories[5], os.listdir(directories[5])[test_uninfected_img_index])



train_parasite_img = cv2.imread(train_parasite_img_path)

dev_parasite_img = cv2.imread(dev_parasite_img_path)

test_parasite_img = cv2.imread(test_parasite_img_path)

train_uninfected_img = cv2.imread(train_uninfected_img_path)

dev_uninfected_img = cv2.imread(dev_uninfected_img_path)

test_uninfected_img = cv2.imread(test_uninfected_img_path)



plt.figure(figsize = (25, 10))



plt.subplot(2,3,1)

plt.imshow(train_parasite_img)

plt.xlabel('Image From ' + directories[0])



plt.subplot(2,3,2)

plt.imshow(dev_parasite_img)

plt.xlabel('Image From ' + directories[1])



plt.subplot(2,3,3)

plt.imshow(test_parasite_img)

plt.xlabel('Image From ' + directories[2])



plt.subplot(2,3,4)

plt.imshow(train_uninfected_img)

plt.xlabel('Image From ' + directories[3])



plt.subplot(2,3,5)

plt.imshow(dev_uninfected_img)

plt.xlabel('Image From ' + directories[4])



plt.subplot(2,3,6)

plt.imshow(test_uninfected_img)

plt.xlabel('Image From ' + directories[5])



plt.show()
from keras import models

from keras import layers

from keras import optimizers

from keras import regularizers

from keras.applications.resnet50 import ResNet50



input_shape = np.array((64, 64, 3))

learning_rate = 0.001

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



def build_pretrained_model():

    inputs = layers.Input(input_shape)

    model = models.Sequential()

    model.add(ResNet50(include_top = False, weights = resnet_weights_path, input_tensor = inputs, pooling = 'max'))

    model.add(layers.Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))

    model.add(layers.Dense(1, activation = 'sigmoid', kernel_regularizer = regularizers.l2(0.01)))

    model.compile(optimizer = optimizers.RMSprop(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])

    

    return model
def build_model(metrics = ['accuracy'], wrapper = None):

    model = models.Sequential()

    model.add(layers.Conv2D(64, 3, activation = 'relu', input_shape = input_shape))

    model.add(layers.MaxPooling2D(2))

    model.add(layers.Conv2D(128, 3, activation = 'relu'))

    model.add(layers.MaxPooling2D(2))

    model.add(layers.Conv2D(256, 3, activation = 'relu'))

    model.add(layers.MaxPooling2D(2))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))

    model.add(layers.Dense(1, activation = 'sigmoid', kernel_regularizer = regularizers.l2(0.01)))

    

    if wrapper:

        loss = wrapper

    else:

        loss = 'binary_crossentropy'

        

    model.compile(optimizer = optimizers.RMSprop(lr = learning_rate), loss = loss, metrics = metrics)

    

    return model
import keras.backend as K



def compute_specificity(y_pred, y_true):

    

    neg_y_true = 1 - y_true

    neg_y_pred = 1 - y_pred

    fp = K.sum(neg_y_true * y_pred)

    tn = K.sum(neg_y_true * neg_y_pred)

    specificity = tn / (tn + fp + K.epsilon())

    return specificity



def specificity_loss_wrapper():

    def specificity_loss(y_true, y_pred):

        return 1.0 - compute_specificity(y_true, y_pred)

    

    return specificity_loss
from keras.preprocessing.image import ImageDataGenerator



train_dir = '../dataset/train/'

dev_dir = '../dataset/dev/'

test_dir = '../dataset/test/'



train_datagen = ImageDataGenerator(rescale = 1./255)

dev_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)



train_gen = train_datagen.flow_from_directory(train_dir,

                                              target_size = (64, 64),

                                              batch_size = 128,

                                              class_mode = 'binary')



dev_gen = dev_datagen.flow_from_directory(dev_dir,

                                          target_size = (64, 64),

                                          batch_size = 128,

                                          class_mode = 'binary')



test_gen = test_datagen.flow_from_directory(test_dir,

                                            target_size = (64, 64),

                                            batch_size = 128,

                                            class_mode = 'binary')



train_steps_per_epoch = train_gen.n // train_gen.batch_size

dev_steps_per_epoch = dev_gen.n // dev_gen.batch_size

def plot_metrics(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(len(acc))



    plt.subplots(figsize = (25, 10))



    plt.subplot(1,2,1)

    plt.plot(epochs, acc, label = 'Training Accuracy')

    plt.plot(epochs, val_acc, label = 'Validation Accuracy')

    plt.title('Training vs. Validation Accuracy')

    plt.xlabel('epochs')

    plt.ylabel('loss')

    plt.legend()



    plt.subplot(1,2,2)

    plt.plot(epochs, loss, label = 'Training Loss')

    plt.plot(epochs, val_loss, label = 'Validation Loss')

    plt.title('Training vs. Validation Loss')

    plt.xlabel('epochs')

    plt.ylabel('loss')

    plt.legend()



    plt.show()

    

    return
from sklearn.metrics import accuracy_score



def get_accuracy_score(predictions):

    accuracy = predictions[1]

    return 'Accuracy Score: {:0.2f}'.format(accuracy * 100)
res_model = build_pretrained_model()

res_model.summary()
history = res_model.fit_generator(train_gen,

                                  steps_per_epoch = train_steps_per_epoch,

                                  validation_data = dev_gen,

                                  validation_steps = dev_steps_per_epoch,

                                  epochs = 15)



plot_metrics(history)
predictions = res_model.evaluate_generator(test_gen)

print(get_accuracy_score(predictions))
model = build_model()

model.summary()
history = model.fit_generator(train_gen,

                              steps_per_epoch = train_steps_per_epoch,

                              validation_data = dev_gen,

                              validation_steps = dev_steps_per_epoch,

                              epochs = 15)



plot_metrics(history)
predictions = model.evaluate_generator(test_gen)

print(get_accuracy_score(predictions))