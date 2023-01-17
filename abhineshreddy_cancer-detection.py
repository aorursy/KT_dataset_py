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
    #print(f"# Unlabelled Images -> {len(os.listdir('/kaggle/input/Images/Unlabelled Images'))}")
    print(f"# Train Images -> {len(os.listdir('/kaggle/input/Images/Train Images/pCR'))}, {len(os.listdir('/kaggle/input/Images/Train Images/non-pCR'))}")
    print(f"# Validation Images -> {len(os.listdir('/kaggle/input/Images/Validation Images/pCR'))}, {len(os.listdir('/kaggle/input/Images/Validation Images/non-pCR'))}")
    print(f"# Test Images -> {len(os.listdir('/kaggle/input/Images/Test Images/pCR'))}, {len(os.listdir('/kaggle/input/Images/Test Images/non-pCR'))}")



print(f"Before Balancing ::\n{'='*60}")
show_counts()

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
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from keras.models import Model,load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(64, 3, 3, input_shape = (224, 224, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding a second convolutional layer
classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
base_path = '/kaggle/input/Images'

# Create Train and Test Data Generators
train_datagen = ImageDataGenerator(
                 featurewise_center=False,
                 featurewise_std_normalization=False,
                 rotation_range=40,
                 shear_range=0.3,
                 zoom_range=0.2,
                 width_shift_range=0.2,
                 height_shift_range=0.2,
                 fill_mode="nearest",
                horizontal_flip=True,
                vertical_flip=True,
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
check_point = ModelCheckpoint('best_model.hdf5', verbose=1, monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=50, mode='max')

history = classifier.fit_generator(train_generator,
                         samples_per_epoch = 8000,
                         nb_epoch = 50,
                         validation_data = validation_generator,
                         nb_val_samples = 5000,callbacks=[check_point, early_stopping])
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
test_generator = validation_datagen.flow_from_directory(f'{base_path}/Test Images',
                                                  target_size=(224,224),
                                                  color_mode='rgb',
                                                  batch_size=20,
                                                  class_mode='binary',
                                                  shuffle=False)

y_true = test_generator.classes

best_model = load_model('best_model.hdf5')
y_pred = best_model.predict_generator(generator=test_generator, verbose=1)
y_hat = y_pred > 0.33
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
    pred_label = 'non-pCR' if pred[0][0] < 0.33 else 'pCR'
    print(f"Actual Label -> {label}")
    print(f"Predicted Label -> {pred_label}")
    print(f"Prediction Score -> {pred[0][0]}")
    
display_pred('QIN-BREAST-01-0059_000.jpg', 'non-pCR')
display_pred('QIN-BREAST-01-0058_005.jpg', 'pCR')