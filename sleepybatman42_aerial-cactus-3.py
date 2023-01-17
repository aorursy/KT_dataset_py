import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, cv2, sys

from IPython.display import Image

from keras.preprocessing import image

from keras import optimizers

from keras import layers,models

from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

import seaborn as sns

import keras

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16

from keras.models import Sequential

from keras.layers import *



print(os.listdir("../input/aerial-cactus-identification"))



from zipfile import ZipFile

import zipfile



import math

import numpy as np

import shutil
test = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")

train = pd.read_csv("../input/aerial-cactus-identification/train.csv")
#Loading all images from the training zip file

zipref = zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/train.zip')



zipref.extractall()
#Loading all images from the test zip file

zipref = zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/test.zip')



zipref.extractall()
#Setting training and test directories

train_dir = "train/"

test_dir = "test/"
#17500 images in the training file

len(os.listdir('train/'))
#4000 images in the test file

len(os.listdir('test/'))
train.head()
test.head()
train.has_cactus=train.has_cactus.astype(str)
print('Aerial Cactus dataset has {} rows and {} columns'.format(train.shape[0],train.shape[1]))
train['has_cactus'].value_counts()
len(train[train['has_cactus'] == '1'])
cactus_train_count = len(train[train['has_cactus'] == '1'])

total_count = len(train['has_cactus'])
Cactus_Proportion = print("There are ", round(cactus_train_count/total_count * 100), "% cacti in the train dataset.")

# In 75% of photos there is a cactus 
#Pie chart to show the percentage from above

value_counts = train.has_cactus.value_counts()

%matplotlib inline

plt.pie(value_counts, labels=['Has Cactus', 'No Cactus'], autopct='%1.1f', colors=['green', 'red'], shadow=True)

plt.figure(figsize=(5,5))

plt.show()
train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.20)

        

test_datagen = ImageDataGenerator(rescale=1/255) 

        

bs = 64
train_generator = train_datagen.flow_from_dataframe(

    dataframe = train,

    directory = train_dir,

    x_col = "id",

    y_col = "has_cactus",

    subset = "training",

    batch_size = bs,

    shuffle = True,

    class_mode = "categorical",

    target_size = (32,32))
valid_generator = train_datagen.flow_from_dataframe(

    dataframe = train,

    directory = train_dir,

    x_col = "id",

    y_col = "has_cactus",

    subset = "validation",

    batch_size = bs,

    shuffle = True,

    class_mode = "categorical",

    target_size = (32,32))
test_generator = test_datagen.flow_from_dataframe(

    dataframe = test,

    directory = test_dir,

    x_col = "id",

    y_col = None,

    batch_size = bs,

    shuffle = False,

    class_mode = None,

    target_size = (32,32))
tr_size = 14000

va_size = 3500

te_size = 4000

tr_steps = math.ceil(tr_size / bs)

va_steps = math.ceil(va_size / bs)

te_steps = math.ceil(te_size / bs)
def training_images(seed):

    np.random.seed(seed)

    train_generator.reset()

    imgs, labels = next(train_generator)

    tr_labels = np.argmax(labels, axis=1)

    

    plt.figure(figsize=(14,14))

    for i in range(36):

        text_class = labels[i]

        plt.subplot(6,6,i+1)

        plt.imshow(imgs[i,:,:,:])

        if(text_class[0] == 1):

            plt.text(0, -2, 'Negative', color='r')

        else:

            plt.text(0, -2, 'Positive', color='b')

        plt.axis('off')

    plt.show()

    

    

training_images(1)
np.random.seed(1)



cnn = Sequential()



cnn.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', input_shape=(32,32,3)))

cnn.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same'))

cnn.add(MaxPooling2D(2,2))

cnn.add(BatchNormalization())



cnn.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))

cnn.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))

cnn.add(MaxPooling2D(2,2))

cnn.add(BatchNormalization())



cnn.add(Flatten())

cnn.add(Dense(128, activation='relu'))

cnn.add(BatchNormalization())



cnn.add(Dense(2, activation='softmax'))



cnn.summary()
%%time



opt = keras.optimizers.Adam(0.01)

cnn.compile(loss='categorical_crossentropy', optimizer=opt,

            metrics=['accuracy'])



h1 = cnn.fit_generator(train_generator, steps_per_epoch=tr_steps, epochs=100,

                          validation_data=valid_generator, validation_steps=va_steps,

                          verbose=1)
start = 1

ep_rng = np.arange(start, len(h1.history['accuracy']))



plt.figure(figsize=[12,6])

plt.subplot(1,2,1)

plt.plot(ep_rng, h1.history['accuracy'][start:], label='Training Accuracy')

plt.plot(ep_rng, h1.history['val_accuracy'][start:], label='Validation Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.subplot(1,2,2)

plt.plot(ep_rng, h1.history['loss'][start:], label='Training Loss')

plt.plot(ep_rng, h1.history['val_loss'][start:], label='Validation Loss')

plt.xlabel('Epoch')

plt.legend()



plt.show()
keras.backend.set_value(cnn.optimizer.lr, 0.001)



h2 = cnn.fit_generator(train_generator, steps_per_epoch=tr_steps, epochs=30,

                      validation_data=valid_generator, validation_steps=va_steps,

                      verbose=1)
start = 1

ep_rng = np.arange(start, len(h2.history['accuracy']))



plt.figure(figsize=[12,6])

plt.subplot(1,2,1)

plt.plot(ep_rng, h2.history['accuracy'][start:], label='Training Accuracy')

plt.plot(ep_rng, h2.history['val_accuracy'][start:], label='Validation Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.subplot(1,2,2)

plt.plot(ep_rng, h2.history['loss'][start:], label='Training Loss')

plt.plot(ep_rng, h2.history['val_loss'][start:], label='Validation Loss')

plt.xlabel('Epoch')

plt.legend()



plt.show()
start = 1



tr_acc = h1.history['accuracy'] + h2.history['accuracy']

va_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']

tr_loss = h1.history['loss'] + h2.history['loss']

va_loss = h1.history['val_loss'] + h2.history['val_loss']



ep_rng = np.arange(start,len(tr_acc))



plt.figure(figsize=[12,6])

plt.subplot(1,2,1)

plt.plot(ep_rng, tr_acc[start:], label='Training Accuracy')

plt.plot(ep_rng, va_acc[start:], label='Validation Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.subplot(1,2,2)

plt.plot(ep_rng, tr_loss[start:], label='Training Loss')

plt.plot(ep_rng, va_loss[start:], label='Validation Loss')

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
shutil.rmtree('/kaggle/working/train')

shutil.rmtree('/kaggle/working/test')