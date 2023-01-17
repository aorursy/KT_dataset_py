import numpy as np # linear algebra

import random

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import cv2

import shutil

from glob import glob

# Helper libraries

import matplotlib.pyplot as plt

import math

%matplotlib inline

print(tf.__version__)
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import cv2



import tensorflow as tf



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping

import os



import sklearn.metrics
dire = '/kaggle/input/covidct/'

subdire_pos = os.path.join('/kaggle/input/covidct/CT_COVID/')

subdire_neg = os.path.join('/kaggle/input/covidct/CT_NonCOVID/')

print(f'Positive samples: {len(os.listdir(subdire_pos))}.')

print(f'Negative samples: {len(os.listdir(subdire_neg))}.')
# jpg and png files

positive_images_ls = glob(os.path.join(subdire_pos,"*.png"))



negative_images_ls = glob(os.path.join(subdire_neg,"*.png"))

negative_images_ls.extend(glob(os.path.join(subdire_neg,"*.jpg")))
covid = {'class': 'CT_COVID',

         'path': subdire_pos,

         'images': positive_images_ls}



non_covid = {'class': 'CT_NonCOVID',

             'path': subdire_neg,

             'images': negative_images_ls}
total_positive_covid = len(positive_images_ls)

total_negative_covid = len(negative_images_ls)

print("Total Positive Cases Covid19 images: {}".format(total_positive_covid))

print("Total Negative Cases Covid19 images: {}".format(total_negative_covid))
img_positive = cv2.imread(os.path.join(positive_images_ls[1]))

img_negative = cv2.imread(os.path.join(negative_images_ls[5]))



f = plt.figure(figsize=(8, 8))

f.add_subplot(1, 2, 1)

plt.imshow(img_negative)

f.add_subplot(1,2, 2)

plt.imshow(img_positive)
print("Image COVID Shape {}".format(img_positive.shape))

print("Image Non COVID Shape {}".format(img_negative.shape))
# Create Train-Test Directory

subdirs  = ['train/', 'test/']

for subdir in subdirs:

    labeldirs = ['CT_COVID', 'CT_NonCOVID']

    for labldir in labeldirs:

        newdir = subdir + labldir

        os.makedirs(newdir, exist_ok=True)
# seed random number generator

random.seed(123)

test_ratio = 0.1





for cases in [covid, non_covid]:

    total_cases = len(cases['images']) #number of total images

    num_to_select = int(test_ratio * total_cases) #number of images to copy to test set

    

    print(cases['class'], num_to_select)

    

    list_of_random_files = random.sample(cases['images'], num_to_select) #random files selected



    for files in list_of_random_files:

        shutil.copy2(files, 'test/' + cases['class'])
# Copy Images to train set

for cases in [covid, non_covid]:

    image_test_files = os.listdir('test/' + cases['class']) # list test files 

    for images in cases['images']:

        if images.split('/')[-1] not in (image_test_files): #exclude test files from shutil.copy

            shutil.copy2(images, 'train/' + cases['class'])
total_train_covid = len(os.listdir('/kaggle/working/train/CT_COVID'))

total_train_noncovid = len(os.listdir('/kaggle/working/train/CT_NonCOVID'))

total_test_covid = len(os.listdir('/kaggle/working/test/CT_COVID'))

total_test_noncovid = len(os.listdir('/kaggle/working/test/CT_NonCOVID'))



print("Train sets images COVID: {}".format(total_train_covid))

print("Train sets images Non COVID: {}".format(total_train_noncovid))

print("Test sets images COVID: {}".format(total_test_covid))

print("Test sets images Non COVID: {}".format(total_test_noncovid))
batch_size = 64

epochs = 10

IMG_HEIGHT = 150

IMG_WIDTH = 150
train_image_generator = ImageDataGenerator(rescale=1./255) 

test_image_generator = ImageDataGenerator(rescale=1./255) 
train_dir = os.path.join('/kaggle/working/train')

test_dir = os.path.join('/kaggle/working/test')





total_train = total_train_covid + total_train_noncovid

total_test = total_test_covid + total_test_noncovid
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,

                                                           directory=train_dir,

                                                           shuffle=True,

                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                           class_mode='binary')
test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,

                                                              directory=test_dir,

                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                              class_mode='binary')
model = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1)

])
model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=total_train // batch_size,

    epochs=epochs,

    validation_data=test_data_gen,

    validation_steps=total_test // batch_size

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss=history.history['loss']

val_loss=history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
import sklearn.metrics

y_pred = (model.predict_generator(test_data_gen) > 0.5).astype(int)

y_true = test_data_gen.classes



for name, value in zip(model.metrics_names, model.evaluate_generator(test_data_gen)):

    print(f'{name}: {value}')

    

print(f'F1 score: {sklearn.metrics.f1_score(y_true, y_pred)}')
pd.DataFrame(sklearn.metrics.confusion_matrix(y_true, y_pred), 

             columns=['pred no covid', 'pred covid'], 

             index=['true no covid', 'true covid'])
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

model.save_weights("model.h5")