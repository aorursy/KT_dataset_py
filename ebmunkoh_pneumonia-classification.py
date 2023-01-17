# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pneumonia_path = '../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person103_bacteria_489.jpeg'

normal_path = '../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0005-0001.jpeg'
pneumonia_path
import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import seaborn as sns



img = mpimg.imread(pneumonia_path)

#imgplot = plt.imshow(img, cmap = 'gray')

imgplot = plt.imshow(img)
img = mpimg.imread(normal_path)

imgplot = plt.imshow(img)
#import tensorflow

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (64,64)

datagen = ImageDataGenerator(samplewise_center = True,

                             samplewise_std_normalization = True,

                             horizontal_flip = True,

                             vertical_flip = True,

                             height_shift_range = 0.05,

                             width_shift_range = 0.1,

                             rotation_range = 5,

                             shear_range = 0.1,

                             fill_mode = 'reflect',

                             zoom_range = 0.15)
train_generator = datagen.flow_from_directory(

        '../input/chest-xray-pneumonia/chest_xray/train',

        target_size=IMG_SIZE,

        color_mode = 'grayscale',

        batch_size=32,

        class_mode='binary'

        )



x_val, y_val = next(datagen.flow_from_directory(

        '../input/chest-xray-pneumonia/chest_xray/val',

        target_size=IMG_SIZE,

        color_mode = 'grayscale',

        batch_size=32,

        class_mode='binary')) # one big batch



x_test, y_test = next(datagen.flow_from_directory(

        '../input/chest-xray-pneumonia/chest_xray/test',

        target_size=IMG_SIZE,

        color_mode = 'grayscale',

        batch_size=180,

        class_mode='binary')) # one big batch
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Dense, Flatten



model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=x_test.shape[1:]))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy',

                           metrics = ['binary_accuracy', 'mae'])

model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('pneumonia_cnn')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)



early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=5)

callbacks_list = [checkpoint, early]
#First Round 

model.fit_generator(train_generator, 

                    steps_per_epoch=100, 

                    validation_data = (x_val, y_val), 

                    epochs = 10, 

                    callbacks = callbacks_list)



   

# Save the entire model as a SavedModel

model.save('pneumonia_cnn') 
scores = model.evaluate(x_test, y_test)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print("val_loss:", scores[0])

print("val_mean_absolute_error:", scores[2])
pred_Y = model.predict(x_test, batch_size = 32, verbose = True)

print(pred_Y[:15])
print(y_test[:15])
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class



num_classes = 0



fpr = dict()

tpr = dict()

roc_auc = dict()

fpr, tpr, _ = roc_curve(y_test, pred_Y)

roc_auc = auc(fpr, tpr)



plt.figure(figsize=(11,8))

lw = 2

plt.plot(fpr, tpr, color='darkorange', 

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('roc2.png')