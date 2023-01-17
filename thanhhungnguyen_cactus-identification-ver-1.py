import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import math



import keras

from keras.models import Sequential

from keras.layers import *

from keras_preprocessing.image import ImageDataGenerator



import zipfile 
train_data = pd.read_csv("../input/aerial-cactus-identification/train.csv", dtype=str)

test_data = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv", dtype=str)
train_data.head()
train_data.describe()
test_data.describe()
zip_ref_1 = zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/test.zip')

zip_ref_1.extractall()
zip_ref_2 = zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/train.zip')

zip_ref_2.extractall()
train_path = "train/"

test_path = "test/"

print('Training Images:', len(os.listdir(train_path)))

print('Testing Images: ', len(os.listdir(test_path)))
train_datagen = ImageDataGenerator(rescale= 1/255, validation_split = 0.20)

test_datagen = ImageDataGenerator(rescale = 1/255)
bs = 100



train_generator = train_datagen.flow_from_dataframe(

    dataframe = train_data,

    directory = train_path,

    x_col = "id",

    y_col = "has_cactus",

    subset = "training",

    batch_size = bs,

    shuffle = True,

    class_mode = "categorical",

    target_size = (32,32))



valid_generator = train_datagen.flow_from_dataframe(

    dataframe = train_data,

    directory = train_path,

    x_col = "id",

    y_col = "has_cactus",

    subset = "validation",

    batch_size = bs,

    shuffle = True,

    class_mode = "categorical",

    target_size = (32,32))



test_generator = test_datagen.flow_from_dataframe(

    dataframe = test_data,

    directory = test_path,

    x_col = "id",

    y_col = None,

    batch_size = bs,

    seed = 1,

    shuffle = False,

    class_mode = None,

    target_size = (32,32))
tr_size = 14000

va_size = 3500

te_size = 4000

tr_steps = math.ceil(tr_size/bs)

va_steps = math.ceil(va_size/bs)

te_steps = math.ceil(te_size/bs)
cnn = Sequential()



cnn.add(Conv2D(28, (3,3), activation = 'relu', padding = 'same', input_shape = (32,32,3)))

cnn.add(Conv2D(28, (3,3), activation = 'relu', padding = 'same'))

cnn.add(MaxPooling2D(2,2))

cnn.add(BatchNormalization())



cnn.add(Conv2D(56, (3,3), activation = 'relu', padding = 'same'))

cnn.add(Conv2D(56, (3,3), activation = 'relu', padding = 'same'))

cnn.add(MaxPooling2D(2,2))

cnn.add(BatchNormalization())



cnn.add(Flatten())

cnn.add(Dense(128, activation = 'relu'))

cnn.add(BatchNormalization())



cnn.add(Dense(2, activation = "softmax"))



cnn.summary()
opt = keras.optimizers.Adam(0.001)

cnn.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])



h1 = cnn.fit_generator(train_generator, steps_per_epoch = tr_steps, epochs=80, validation_data = valid_generator, validation_steps=va_steps, verbose=1)
start = 1

ep_rng = np.arange(start,len(h1.history['accuracy']))



plt.figure(figsize = [12,6])

plt.subplot(1,2,1)

plt.plot(ep_rng, h1.history['accuracy'][start:], label = 'Training Accuracy')

plt.plot(ep_rng, h1.history['val_accuracy'][start:], label = 'Validation Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.subplot(1,2,2)

plt.plot(ep_rng, h1.history['loss'][start:], label = 'Training Loss')

plt.plot(ep_rng, h1.history['val_loss'][start:], label = 'Validation Loss')

plt.xlabel('Epoch')

plt.legend()



plt.show()
test_pred = cnn.predict_generator(test_generator, steps=te_steps, verbose=1)
test_fnames = test_generator.filenames

pred_classes = np.argmax(test_pred, axis=1)



print(np.sum(pred_classes == 0))

print(np.sum(pred_classes == 1))
submission = pd.DataFrame({

    'id':test_fnames,

    'has_cactus':pred_classes

})



submission.to_csv('submission.csv', index=False)



submission.head()
import shutil



shutil.rmtree('/kaggle/working/train')

shutil.rmtree('/kaggle/working/test')