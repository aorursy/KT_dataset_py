import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from PIL import Image

import random



import seaborn as sns # for making plots with seaborn

color = sns.color_palette()



import matplotlib.pyplot as plt



# Networks

from keras.preprocessing import image

from keras.applications.resnet50 import ResNet50

from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import VGG19

from keras.applications.inception_v3 import InceptionV3

from keras.applications.xception import Xception

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.mobilenet import MobileNet

from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

from keras.applications.nasnet import NASNetLarge, NASNetMobile

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import GlobalAveragePooling2D



# Layers

from keras.layers import Dense, Activation, Flatten, Dropout

from keras import backend as K



# Other

from keras import optimizers

from keras import losses

from keras.optimizers import SGD, Adam

from keras.models import Sequential, Model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping

from keras.models import load_model



# Utils

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random, glob

import os, sys, csv

import cv2

import time, datetime

from sklearn.utils import class_weight



# Files

import utils



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Set random seed for reprodue results

def seed_everything(seed=1234):

    random.seed(seed)

    np.random.seed(seed)

    

seed_everything()
train = pd.read_csv("../input/aerial-cactus-identification/train.csv")

print(f'The train dataset have {train.shape[0]} rows and {train.shape[1]} columns')
train['has_cactus']=train['has_cactus'].astype(str)

train.head()
sns.countplot(train['has_cactus'])
# train_validation split

training_data_percent = 0.85

len_df=len(train.id)

TRAINING_SAMPLE=int(len_df*training_data_percent)

VALIDATION_SAMPLE = int(len_df-TRAINING_SAMPLE)



print(f'The no. of training samples are {TRAINING_SAMPLE} and we are taking {training_data_percent * 100}% data as training\n Validation samples: {VALIDATION_SAMPLE} ')
# FILE_PATH

TRAIN_DIR = "../input/aerial-cactus-identification/train/train/"

TEST_DIR = "../input/aerial-cactus-identification/test/test"

WEIGHT_FILE = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'



# GENERAL

DROPOUT_RATE      = 0.4

NB_CLASSES        = 2



# LEARNING

BATCH_SIZE        = 128

NB_EPOCHS_BOTTLENECK = 5

NB_EPOCHS_FINETUNING = 5



# Global settings

model = "VGG16"

WIDTH, HEIGHT = 331,331

FC_LAYERS = [1024]
class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(train['has_cactus']),

                                                 train['has_cactus'])



print(f'The class weight of class0 i.e No cactus is {class_weights[0]} and \nClass weight of class1 i.e Has cactus is {class_weights[1]}')
# This function prepares a random batch from the dataset

def load_batch(dataset_df, batch_size = 25):

    batch_df = dataset_df.loc[np.random.permutation(np.arange(0,

                                                              len(dataset_df)))[:batch_size],:]

    return batch_df



# This function plots sample images in specified size and in defined grid

def plot_batch(images_df, grid_width, grid_height, im_scale_x, im_scale_y):

    f, ax = plt.subplots(grid_width, grid_height)

    f.set_size_inches(12, 12)



    img_idx = 0

    for i in range(0, grid_width):

        for j in range(0, grid_height):

            ax[i][j].axis('off')

            ax[i][j].set_title(images_df.iloc[img_idx]['has_cactus'])

            ax[i][j].imshow(Image.open(TRAIN_DIR + images_df.iloc[img_idx]['id']).resize

                                             ((im_scale_x,im_scale_y)))

            img_idx += 1

            

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.25)
batch_df = load_batch(train, 

                    batch_size=36)
plot_batch(batch_df, grid_width=6, grid_height=6

           ,im_scale_x=64, im_scale_y=64)
datagen = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')



val_datagen = ImageDataGenerator(rescale=1./255)



train_generator=datagen.flow_from_dataframe(dataframe=train[:TRAINING_SAMPLE],directory=TRAIN_DIR,x_col='id',

                                            y_col='has_cactus',class_mode='binary',batch_size=BATCH_SIZE,

                                            target_size=(WIDTH,HEIGHT))





validation_generator= val_datagen.flow_from_dataframe(dataframe=train[TRAINING_SAMPLE-1:len_df],directory=TRAIN_DIR,x_col='id',

                                                y_col='has_cactus',class_mode='binary',batch_size=BATCH_SIZE,

                                                target_size=(WIDTH,HEIGHT))
# Prepare the model

if model == "VGG16":

    from keras.applications.vgg16 import preprocess_input

    preprocessing_function = preprocess_input

    base_model = VGG16(weights = WEIGHT_FILE, include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "VGG19":

    from keras.applications.vgg19 import preprocess_input

    preprocessing_function = preprocess_input

    base_model = VGG19(weights = WEIGHT_FILE, include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "ResNet50":

    from keras.applications.resnet50 import preprocess_input

    preprocessing_function = preprocess_input

    base_model = ResNet50(weights = WEIGHT_FILE, include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "InceptionV3":

    from keras.applications.inception_v3 import preprocess_input

    preprocessing_function = preprocess_input

    base_model = InceptionV3(weights = WEIGHT_FILE, include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "Xception":

    from keras.applications.xception import preprocess_input

    preprocessing_function = preprocess_input

    base_model = Xception(weights = WEIGHT_FILE, include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "InceptionResNetV2":

    from keras.applications.inceptionresnetv2 import preprocess_input

    preprocessing_function = preprocess_input

    base_model = InceptionResNetV2(weights = WEIGHT_FILE, include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "MobileNet":

    from keras.applications.mobilenet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = MobileNet(weights = WEIGHT_FILE, include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "DenseNet121":

    from keras.applications.densenet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = DenseNet121(weights = WEIGHT_FILE, include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "DenseNet169":

    from keras.applications.densenet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = DenseNet169(weights = WEIGHT_FILE, include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "DenseNet201":

    from keras.applications.densenet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = DenseNet201(weights = WEIGHT_FILE, include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "NASNetLarge":

    from keras.applications.nasnet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = NASNetLarge(weights = WEIGHT_FILE, include_top=True, input_shape=(HEIGHT, WIDTH, 3))

elif model == "NASNetMobile":

    from keras.applications.nasnet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = NASNetMobile(weights = WEIGHT_FILE, include_top=False, input_shape=(HEIGHT, WIDTH, 3))

else:

    ValueError("The model you requested is not supported in Keras")

    

# Add on new FC layers with dropout for intializing the final fully connected layer



def build_bottleneck_model(base_model, dropout, fc_layers, num_classes):

    # Freeze All Layers Except Bottleneck Layers for Fine-Tuning

    for layer in base_model.layers:

        layer.trainable = False



    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    

    for fc in fc_layers:

        x = Dense(fc, activation='relu')(x) # New FC layer, random init

        x = Dropout(dropout)(x)



    predictions = Dense(num_classes-1, activation='sigmoid')(x)

    

    bottleneck_model = Model(inputs=base_model.input, outputs=predictions)



    return bottleneck_model
# Add on new FC layers with dropout for fine tuning



def build_finetuning_model(base_model, dropout, fc_layers, num_classes):



    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    

    for fc in fc_layers:

        x = Dense(fc, activation='relu')(x) # New FC layer, random init

        x = Dropout(dropout)(x)



    predictions = Dense(num_classes-1, activation='sigmoid')(x)

    

    finetuning_model = Model(inputs=base_model.input, outputs=predictions)



    return finetuning_model
bottleneck_model = build_bottleneck_model(base_model, DROPOUT_RATE, FC_LAYERS, NB_CLASSES)

adam = Adam(lr=0.001)

bottleneck_model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])

    

    

def lr_decay(epoch):

    if epoch%3 == 0 and epoch!=0:

        lr = K.get_value(model.optimizer.lr)

        K.set_value(model.optimizer.lr, lr/2)

        print("LR changed to {}".format(lr/2))

    return K.get_value(model.optimizer.lr)



learning_rate_schedule = LearningRateScheduler(lr_decay)



early_stopping = EarlyStopping(patience=2)



filepath= "../working/" + "weightfile.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

callbacks_list = [checkpoint,early_stopping]





history = bottleneck_model.fit_generator(train_generator, epochs=NB_EPOCHS_BOTTLENECK, workers=8, steps_per_epoch= TRAINING_SAMPLE // BATCH_SIZE, 

validation_data=validation_generator, validation_steps= VALIDATION_SAMPLE // BATCH_SIZE, class_weight=class_weights, shuffle=True, callbacks=callbacks_list)
# Plot the training and validation loss + accuracy

def plot_training(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(len(acc))



    plt.plot(epochs, acc, 'r.')

    plt.plot(epochs, val_acc, 'r')

    plt.title('Training and validation accuracy')



    # plt.figure()

    # plt.plot(epochs, loss, 'r.')

    # plt.plot(epochs, val_loss, 'r-')

    # plt.title('Training and validation loss')

    plt.show()
plot_training(history)
weights_file_path = '../working/weightfile.h5'
finetuning_model = build_finetuning_model(base_model, DROPOUT_RATE, FC_LAYERS, NB_CLASSES)

finetuning_model.load_weights(weights_file_path)

finetuning_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])



# Freeze Half of all Layers Except final Eight layers of VGG_NET Fine-Tuning

for layer in finetuning_model.layers[:8]:

    layer.trainable = False

for layer in finetuning_model.layers[8:]:

    layer.trainable = True

    

    

optimizer=SGD(lr=0.001, momentum=0.9)    

def lr_decay(epoch):

    if epoch%3 == 0 and epoch!=0:

        lr = K.get_value(model.optimizer.lr)

        K.set_value(model.optimizer.lr, lr/2)

        print("LR changed to {}".format(lr/2))

    return K.get_value(model.optimizer.lr)



# learning_rate_schedule = LearningRateScheduler(lr_decay)



early_stopping = EarlyStopping(patience=2)



filepath= "../working/" + "_weightfile_finetuning.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

callbacks_list = [checkpoint,early_stopping]





history_finetuning = bottleneck_model.fit_generator(train_generator, epochs=NB_EPOCHS_FINETUNING, workers=8, steps_per_epoch= TRAINING_SAMPLE // BATCH_SIZE, 

validation_data=validation_generator, validation_steps= VALIDATION_SAMPLE // BATCH_SIZE, class_weight=class_weights, shuffle=True, callbacks=callbacks_list)
plot_training(history_finetuning)