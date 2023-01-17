import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from IPython.display import Image

import matplotlib.pyplot as plt

import seaborn as sns

import math



import keras

from keras.models import Sequential

from keras.layers import Conv2D #MaxPooling2D, Flatten,Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import *

from keras import layers,models



import zipfile 
print(os.listdir("../input/aerial-cactus-identification"))
train = pd.read_csv('../input/aerial-cactus-identification/train.csv', dtype=str)

test = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv', dtype=str)
print(train.head())

print(test.head())
print('out dataset has {} rows and {} columns'.format(train.shape[0], train.shape[1]))
train['has_cactus'].value_counts()
fig = plt.figure(figsize = (6,4)) 

ax = sns.countplot(train.has_cactus).set_title('Has Cactus Counts', fontsize = 18)

plt.annotate(train.has_cactus.value_counts()[0],

            xy = (1, train.has_cactus.value_counts()[0]),

            va = 'bottom',

            ha = 'center',

            fontsize = 12)

plt.annotate(train.has_cactus.value_counts()[1],

            xy = (0, train.has_cactus.value_counts()[1]),

            va = 'bottom',

            ha = 'center',

            fontsize = 12)

plt.ylim(0,16000)

plt.ylabel('Count', fontsize = 16)

plt.xlabel('Labels', fontsize = 16)

plt.show()
labels_count = train.has_cactus.value_counts()



%matplotlib inline

plt.pie(labels_count, labels=['Cactus', 'No Cactus'], startangle=180, 

        autopct='%1.1f', colors=['#00ff99','#FF96A7'], shadow=True)

plt.figure(figsize=(16, 16))

plt.show()
zip_ref_1 = zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/test.zip')

zip_ref_1.extractall()
zip_ref_2 = zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/train.zip')

zip_ref_2.extractall()
train_dir = "train/"

test_dir = "test/"

print('Training Images:', len(os.listdir(train_dir)))

print('Testing Images: ', len(os.listdir(test_dir)))
train_datagen = ImageDataGenerator(rescale = 1/255, validation_split=0.20, horizontal_flip=True, vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1/255, horizontal_flip=True, vertical_flip=True)
batch_size = 64



#This will use 80% of the data

train_generator = train_datagen.flow_from_dataframe(

        dataframe = train, 

        directory = train_dir, 

        x_col = 'id',

        y_col = 'has_cactus', 

        subset = 'training',

        batch_size = batch_size,

        shuffle = True,

        #class_mode = 'binary',

        class_mode = 'categorical',

        target_size = (32, 32))



#This will use the 20% of the data

val_generator = train_datagen.flow_from_dataframe(

        dataframe = train, 

        directory = train_dir, 

        x_col = 'id',

        y_col = 'has_cactus',  

        subset = 'validation',

        batch_size = batch_size,

        shuffle = True,

        #class_mode = 'binary',

        class_mode = 'categorical',

        target_size = (32, 32))



test_generator = test_datagen.flow_from_dataframe(

        dataframe = test,

        directory = test_dir,

        x_col = "id",

        y_col = None,

        batch_size = batch_size,

        seed = 1,

        shuffle = False,

        class_mode = None,

        target_size = (32, 32))
tr_size = 14000

va_size = 3500

te_size = 4000

tr_steps = math.ceil(tr_size / batch_size)

va_steps = math.ceil(va_size / batch_size)

te_steps = math.ceil(te_size / batch_size)
def training_images(seed):

    np.random.seed(seed)

    train_generator.reset()

    imgs, labels = next(train_generator)

    tr_labels = np.argmax(labels, axis=1)

    

    plt.figure(figsize=(8, 8))

    for i in range(25):

        text_class = labels[i]

        plt.subplot(5, 5, i+1)

        plt.imshow(imgs[i, :, :, :])

        if(text_class[0] == 0):

            plt.text(0, -2, 'Positive', color='r')

        else:

            plt.text(0, -2, 'Negative', color='b')

        plt.axis('off')

    plt.show()



training_images(1)
np.random.seed(1)



cnn=Sequential()



cnn.add(Conv2D(32, (3, 3), activation='relu', padding = 'same', input_shape=(32, 32, 3)))

cnn.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))

cnn.add(MaxPool2D((2, 2)))

cnn.add(BatchNormalization())



cnn.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))

cnn.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))

cnn.add(MaxPool2D((2, 2)))

cnn.add(BatchNormalization())



cnn.add(Flatten())

cnn.add(Dense(128, activation='relu'))

cnn.add(BatchNormalization())

cnn.add(Dense(2, activation='softmax'))

cnn.summary()
%%time 



epochs = 50

opt = keras.optimizers.Adam(learning_rate=0.001)



cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



h1 = cnn.fit_generator(train_generator, steps_per_epoch=tr_steps, epochs=epochs,

                       validation_data=val_generator, validation_steps=va_steps, 

                       verbose=1)
epochs_range = range(1, len(h1.history['accuracy']) + 1)



plt.figure(figsize=[12, 6])

plt.subplot(1, 2, 1)

plt.plot(epochs_range, h1.history['accuracy'], label='Training Accuracy')

plt.plot(epochs_range, h1.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(epochs_range, h1.history['loss'], label='Loss')

plt.plot(epochs_range, h1.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')

plt.legend()



plt.show()
test_pred = cnn.predict_generator(test_generator, steps = te_steps, verbose=1)
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