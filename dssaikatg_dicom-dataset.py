# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import shutil



# base_path = '/kaggle/input/'

# print(os.listdir('/kaggle/input'))



def show_counts():

    print(f"# Unlabelled Images -> {len(os.listdir('/kaggle/input/Images/Unlabelled Images'))}")

    print(f"# Train Images -> {len(os.listdir('/kaggle/input/Images/Train Images/pCR'))}, {len(os.listdir('/kaggle/input/Images/Train Images/non-pCR'))}")

    print(f"# Validation Images -> {len(os.listdir('/kaggle/input/Images/Validation Images/pCR'))}, {len(os.listdir('/kaggle/input/Images/Validation Images/non-pCR'))}")

    print(f"# Test Images -> {len(os.listdir('/kaggle/input/Images/Test Images/pCR'))}, {len(os.listdir('/kaggle/input/Images/Test Images/non-pCR'))}")



def create_balance(path='/kaggle/input/Images/Train Images/pCR'):

    files = os.listdir(path)

    for file in files:

        shutil.copy(f'{path}/{file}', f'{path}/Copy_{file}')

        if len(os.listdir(path)) >= 43206:

            break

    num_copies = len([x for x in os.listdir(path) if x.startswith('Copy_')])

    print(f"No. of Copies created -> {num_copies}")



print(f"Before Balancing ::\n{'='*60}")

show_counts()

# create_balance()

# print(f"After Balancing ::\n{'='*60}")

# show_counts()



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import shutil

import random

import pandas as pd

import numpy as np

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt

import cv2

from PIL import Image

import glob

from tqdm import tqdm

import pickle



import keras

import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.applications.resnet50 import ResNet50

from keras.applications.resnet_v2 import ResNet50V2, ResNet152V2

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout, Input, Conv2D, MaxPooling2D, Flatten

from keras.models import Model,load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint
base_path = '/kaggle/input/Images'



# Create Train and Test Data Generators

train_datagen = ImageDataGenerator(

#                 featurewise_center=False,

#                 featurewise_std_normalization=False,

#                 rotation_range=40,

#                 shear_range=0.3,

#                 zoom_range=0.2,

#                 width_shift_range=0.2,

#                 height_shift_range=0.2,

#                 fill_mode="nearest",

#                 horizontal_flip=True,

#                 vertical_flip=True,

                rescale=1./255)



validation_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(f'{base_path}/Train Images',

                                                    target_size=(224,224),

                                                    color_mode='rgb',

                                                    batch_size=20,

                                                    class_mode='binary',

                                                    shuffle=True)



validation_generator = validation_datagen.flow_from_directory(f'{base_path}/Validation Images',

                                                  target_size=(224,224),

                                                  color_mode='rgb',

                                                  batch_size=20,

                                                  class_mode='binary',

                                                  shuffle=True)



print(train_generator.class_indices)
%%time



# Destroy the current TF graph and create a new one and create callbacks to be used later.

K.clear_session()

check_point = ModelCheckpoint('best_model.hdf5', verbose=1, monitor='val_accuracy', save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='max')



def get_model():

    # Create the Base Model and add layers from the Output of the same.

    base_model = ResNet50V2(weights='imagenet', include_top=False)

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(2048)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)



    x = Dense(1024)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dropout(0.5)(x)



    x = Dense(512)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dropout(0.3)(x)



    preds = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=preds)



    # Make the first 50 layers non-trainable

#     for layer in model.layers[:50]:

#         layer.trainable=False



    return model



# def get_basic_model():

#     inp = Input(shape=(299, 299, 3))

#     x = Conv2D(128, (3, 3), padding="same", activation="relu")(inp)

#     x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)

#     x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)

#     x = MaxPooling2D((2, 2))(x)



#     x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)

#     x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)

#     x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)

#     x = MaxPooling2D((2, 2))(x)



#     x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)

#     x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)

#     x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)

#     x = MaxPooling2D((2, 2))(x)



#     x = Flatten()(x)

#     x = Dense(2048)(x)

#     x = BatchNormalization()(x)

#     x = Activation('relu')(x)

#     x = Dropout(0.5)(x)



#     x = Dense(1024)(x)

#     x = BatchNormalization()(x)

#     x = Activation('relu')(x)

#     x = Dropout(0.5)(x)



#     x = Dense(512)(x)

#     x = BatchNormalization()(x)

#     x = Activation('relu')(x)

#     x = Dropout(0.5)(x)



#     preds = Dense(1, activation='sigmoid')(x)

#     model = Model(inputs=inp, outputs=preds)

#     return model



model = get_model()

# model = get_basic_model()

print(model.summary())



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit_generator(generator=train_generator,

                        validation_data=validation_generator,

                        class_weight={0: 1.0, 1: 2.2},

                        use_multiprocessing=True, workers=2,

                        steps_per_epoch=10,

                        epochs=2, verbose=2,

                        callbacks=[check_point, early_stopping])
# Plot the performance of the model

plt.plot(range(len(history.history['loss'])), history.history['val_loss'], label='val_loss')

plt.plot(range(len(history.history['loss'])), history.history['loss'], label='train_loss')

plt.legend()

plt.ylabel('Loss')

plt.xlabel('Iteration')

plt.savefig('Loss.jpg');
# Plot the performance of the model

plt.plot(range(len(history.history['accuracy'])), history.history['val_accuracy'], label='val_accuracy')

plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'], label='train_accuracy')

plt.legend()

plt.ylabel('Accuracy')

plt.xlabel('Iteration')

plt.savefig('Accuracy.jpg');
%%time

test_generator = validation_datagen.flow_from_directory(f'{base_path}/Test Images',

                                                  target_size=(224,224),

                                                  color_mode='rgb',

                                                  batch_size=20,

                                                  class_mode='binary',

                                                  shuffle=False)



best_model = load_model('best_model.hdf5')

y_true = test_generator.classes

y_pred = best_model.predict_generator(generator=test_generator, verbose=1)

y_hat = y_pred > 0.5

print("Accuracy:", accuracy_score(y_true, y_hat), "\n")

print("F1-Score:", f1_score(y_true, y_hat), "\n")

print("Classification Report:\n", classification_report(y_true, y_hat), "\n")

print("Confusion Matrix:\n", confusion_matrix(y_true, y_hat))
temp_path = '/kaggle/input/Images/Test Images'

def display_pred(file, label):

    plt.figure(figsize=(10, 8))

    plt.imshow(plt.imread(f'{temp_path}/{label}/{file}'))

    plt.xticks([])

    plt.yticks([])

    plt.title(f'{label}/{file}')

    plt.savefig(f'Pred_{label}.jpg')

    plt.show()

    

    image = load_img(f'{temp_path}/{label}/{file}', target_size=(224, 224))

    image = img_to_array(image)

    image *= 1./255

    

    pred = best_model.predict([[image]], batch_size=1)

    pred_label = 'non-pCR' if pred[0][0] < 0.5 else 'pCR'

    print(f"Actual Label -> {label}")

    print(f"Predicted Label -> {pred_label}")

    print(f"Prediction Score -> {pred[0][0]}")

    

display_pred('QIN-BREAST-01-0059_000.jpg', 'non-pCR')

display_pred('QIN-BREAST-01-0058_872.jpg', 'pCR')
# pickle.dump(history, open("history.pkl", "wb"))
# from IPython.display import FileLink

# os.chdir(r'/kaggle/working')

# FileLink(r'best_model.hdf5')

# FileLink(r'history.pkl')

# FileLink(r'Pred_pCR.jpg')