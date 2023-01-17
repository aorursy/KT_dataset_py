# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
INPUT_SHAPE = (232,232,4)
BATCH_SIZE = 32
path_to_train = '/kaggle/input/human-protein-atlas-image-classification/train/'
data = pd.read_csv('/kaggle/input/human-protein-atlas-image-classification/train.csv')
weight_path = '/kaggle/input/dna-first-kaggle/'

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)
from sklearn.model_selection import train_test_split
train_ids, test_ids, train_targets, test_target = train_test_split(
    data['Id'], data['Target'], test_size=0.2, random_state=42)
class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 4
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels
            
    
    def load_image(path, shape):
        R = np.array(Image.open(path+'_red.png'))
        G = np.array(Image.open(path+'_green.png'))
        B = np.array(Image.open(path+'_blue.png'))
        Y = np.array(Image.open(path+'_yellow.png'))

        image = np.stack((R,G,B,Y),-1)
        image = cv2.resize(image, (shape[0], shape[1]))
        image = np.divide(image, 255)
        return image  
                
            
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug
# create train datagen
train_datagen = data_generator.create_train(train_dataset_info, BATCH_SIZE, INPUT_SHAPE, augument=True)
images, labels = next(train_datagen)

fig, ax = plt.subplots(1,4,figsize=(25,5))
for i in range(4):
    ax[i].imshow(images[i])
print('min: {0}, max: {1}'.format(images.min(), images.max()))
images.shape
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import Callback
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import tensorflow as tf
import keras
def DnAInput(img_shape = INPUT_SHAPE) :
    model = Sequential()
    # 230 * 230
    model.add(Conv2D(32, kernel_size=(3,3), activation="relu", kernel_initializer='he_normal', input_shape = img_shape))
    model.add(BatchNormalization())
    # 228 * 228
    model.add(Conv2D(32, kernel_size=(3,3), activation="relu", kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # 226 * 226
    model.add(Conv2D(32, kernel_size=(3,3), activation="relu", kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # 224 * 224
    model.add(Conv2D(3, kernel_size=(3,3), activation="relu", kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    return model
def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
def show_history(history):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('f1')
    ax[1].plot(history.epoch, history.history["f1"], label="Train f1")
    ax[1].plot(history.epoch, history.history["val_f1"], label="Validation f1")
    ax[2].set_title('acc')
    ax[2].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[2].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
input_model = DnAInput()
densenet169 = DenseNet169(include_top = False)
densenet169.summary()
for layer in densenet169.layers:
    layer.trainable = False
x = Conv2D(28,kernel_size=(3,3), activation='relu', kernel_initializer="he_normal")(densenet169.output)
x = BatchNormalization()(x)
flat = GlobalAveragePooling2D()(x)
fc = Dense(28, activation='sigmoid')(flat)
inputToDensnet169 = Model(inputs=densenet169.input, outputs=fc)
inputToDensnet169.summary()
tmp_output = input_model.output
final_output = inputToDensnet169(tmp_output)
DnaNet = Model(inputs = input_model.input, output = final_output)
DnaNet.summary()
DnaNet.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy',f1])
import glob
glob.glob(weight_path + '*')
DnaNet.load_weights(weight_path + 'DnAnet64-10.hdf5')
checkpointer = ModelCheckpoint('DnAnet64-{epoch:02d}_ver2.hdf5',
    verbose=2, save_best_only=False)

train_generator = data_generator.create_train(
    train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=False)
validation_generator = data_generator.create_train(
    train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)
history = DnaNet.fit_generator(
    train_generator,
    steps_per_epoch=int(data.shape[0]/BATCH_SIZE),
    validation_data=next(validation_generator),
    epochs=20, 
    verbose=1,
    callbacks=[checkpointer])
