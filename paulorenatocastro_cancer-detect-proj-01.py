import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import math

from keras_preprocessing.image import ImageDataGenerator

import keras

from keras.models import Sequential

from keras.layers import *

from sklearn.utils import shuffle

import os

import cv2

import matplotlib.patches as patches



import seaborn as sns
train_data = pd.read_csv("../input/histopathologic-cancer-detection/train_labels.csv", dtype=str)

test_data = pd.read_csv("../input/histopathologic-cancer-detection/sample_submission.csv", dtype=str)
train_data['label'].value_counts()
print(train_data.shape)

train_data.head()
print(test_data.shape)

test_data.head()
train_path = "../input/histopathologic-cancer-detection/train/"

test_path = "../input/histopathologic-cancer-detection/test/"

# quick look at the label stats

train_data['label'].value_counts()
fig = plt.figure(figsize = (6,6)) 

ax = sns.countplot(train_data.label).set_title('Label Counts', fontsize = 18)

plt.annotate(train_data.label.value_counts()[0],

            xy = (0,train_data.label.value_counts()[0] + 2000),

            va = 'bottom',

            ha = 'center',

            fontsize = 12)

plt.annotate(train_data.label.value_counts()[1],

            xy = (1,train_data.label.value_counts()[1] + 2000),

            va = 'bottom',

            ha = 'center',

            fontsize = 12)

plt.ylim(0,150000)

plt.ylabel('Count', fontsize = 16)

plt.xlabel('Labels', fontsize = 16)

plt.show()
print('Training Images:', len(os.listdir(train_path)))

print('Testing Images: ', len(os.listdir(test_path)))
train_data.id = train_data.id + '.tif'

test_data.id = test_data.id + '.tif'

print(train_data.head())

print(test_data.head())
train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.20)

test_datagen = ImageDataGenerator(rescale=1/255)
tr_size = 176020

va_size = 44005

te_size = 57458
b_size = 32



tr_steps = math.ceil(tr_size / b_size)

va_steps = math.ceil(va_size / b_size)

te_steps = math.ceil(te_size / b_size)



train_generator = train_datagen.flow_from_dataframe(

    dataframe = train_data,

    directory = train_path,

    x_col = "id",

    y_col = "label",

    subset = "training",

    batch_size = b_size,

    #seed = 1,

    shuffle = True,

    class_mode = "categorical",

    target_size = (96, 96))



valid_generator = train_datagen.flow_from_dataframe(

    dataframe = train_data,

    directory = train_path,

    x_col = "id",

    y_col = "label",

    subset = "validation",

    batch_size = b_size,

    #seed = 1,

    shuffle = True,

    class_mode = "categorical",

    target_size = (96, 96))



test_generator = test_datagen.flow_from_dataframe(

    dataframe = test_data,

    directory = test_path,

    x_col = "id",

    y_col = None,

    batch_size = 32,

    seed = 1,

    shuffle = False,

    class_mode = None,

    target_size = (96, 96))
def training_images(seed):

    np.random.seed(seed)

    train_generator.reset()

    imgs, labels = next(train_generator)

    tr_labels = np.argmax(labels, axis=1)

    

    plt.figure(figsize=(10, 10))

    for i in range(10):

        text_class = labels[i]

        plt.subplot(4, 5, i+1)

        plt.imshow(imgs[i, :, :, :])

        if(text_class[0] == 0):

            plt.text(0, -5, 'Positive', color='r')

        else:

            plt.text(0, -5, 'Negative', color='b')

        plt.axis('off')

    plt.show()



training_images(1)
np.random.seed(1)



# Initialising the CNN

model = Sequential()



# Step 1 - Convolution

model.add(Cropping2D(cropping=((32, 32), (32, 32)), input_shape=(96, 96, 3)))

model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))



# Step 2 - Pooling

model.add(MaxPooling2D(2, 2))

model.add(BatchNormalization())



# Adding a second convolutional layer

model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D(2, 2))



# Step 3 - Flattening

model.add(Flatten())



# Step 4 - Full connection

model.add(Dense(64, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Dense(2, activation = 'softmax'))



model.summary()
#from keras.utils import plot_model

#plot_model(model, to_file='model.png')
%%time 



opt = keras.optimizers.Adam(learning_rate=0.00015, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, decay=0.0)



# Compiling the CNN

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])



hist = model.fit_generator(train_generator, epochs=10, validation_data=valid_generator, 

                           steps_per_epoch=tr_steps, validation_steps=va_steps, verbose=1)
epochs_range = range(1, len(hist.history['accuracy']) + 1)



plt.figure(figsize=[12, 6])

plt.subplot(1, 2, 1)

plt.plot(epochs_range, hist.history['accuracy'], label='Training Accuracy')

plt.plot(epochs_range, hist.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(epochs_range, hist.history['loss'], label='Loss')

plt.plot(epochs_range, hist.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')

plt.legend()



plt.show()
test_pred = model.predict_generator(test_generator, steps = te_steps, verbose=1)



pred_classes = np.argmax(test_pred, axis=1)

 

test_fnames = test_generator.filenames

test_fnames = [x.split('.')[0] for x in test_fnames] 
submission = pd.DataFrame({

    'id':test_fnames,

    'label':pred_classes

})

 

submission.to_csv('submission.csv', index=False)

submission.head()