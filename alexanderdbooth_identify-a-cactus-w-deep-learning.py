# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
train = pd.read_csv("../input/train.csv")

train.head()
from sklearn.model_selection import train_test_split

from tensorflow import keras



X_train, X_val, Y_train, Y_val = train_test_split(train.id, train.has_cactus, test_size=0.2)
import os

from os.path import join



#load training images

catctus_dir = '../input/train/train'



#get full image paths for train/val

train_paths = [join(catctus_dir,filename) for filename in X_train]

val_paths = [join(catctus_dir,filename) for filename in X_val]



train_paths[0:5]
from IPython.display import Image, display

for i, img_path in enumerate(train_paths[0:5]):

    display(Image(img_path))

#yup, those are cacti
from tensorflow.keras.preprocessing.image import load_img, img_to_array



#image size

img_rows, img_cols, image_size = 32, 32, 32



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    output = prep_data(img_array)

    return(output)



#training data has its labels already split out

def prep_data(raw):

    x = raw[:,0:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 3)

    out_x = out_x / 255

    return out_x
train_data = read_and_prep_images(train_paths)

val_data = read_and_prep_images(val_paths)
np.shape(train_data) #14000 train images, 3,500 val images
from tensorflow import keras

#cactus or no

num_classes = 2



train_labels = keras.utils.to_categorical(Y_train, num_classes)

val_labels = keras.utils.to_categorical(Y_val, num_classes)
import matplotlib.pyplot as plt



#view a couple of the training images

for i in range(1,13):

    plt.subplot(3,4,i)

    plt.imshow(train_data[i-1])

#moar cacti
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D



#build the model

cactus_model = Sequential()

cactus_model.add(Conv2D(12, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 3))) #activation layer



#additional learning layers

cactus_model.add(Conv2D(20, kernel_size=(3, 3), padding='valid', activation='relu'))

cactus_model.add(Conv2D(20, kernel_size=(3, 3), padding='valid', activation='relu'))

cactus_model.add(Conv2D(20, kernel_size=(3, 3), padding='valid', activation='relu'))

cactus_model.add(Conv2D(20, kernel_size=(3, 3), padding='valid', activation='relu'))



#final prediction layers

cactus_model.add(Flatten())

cactus_model.add(Dense(100, activation='relu'))

cactus_model.add(Dense(num_classes, activation='softmax'))



#compile the model

cactus_model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])
#initial fit with validation

history = cactus_model.fit(train_data, train_labels,

          batch_size=100,

          epochs=10,

          validation_data = (val_data, val_labels))
#https://www.kaggle.com/pheaboo/simple-cnn-trained-from-scratch

plt.figure(figsize=(15,5))



plt.subplot(141)

plt.plot(history.history['loss'], label='training')

plt.plot(history.history['val_loss'], label='validation')

plt.xlabel('# Epochs')

plt.legend()

plt.ylabel("Loss - Binary Cross Entropy")

plt.title('Loss Evolution')



plt.subplot(142)

plt.plot(history.history['loss'], label='training')

plt.plot(history.history['val_loss'], label='validation')

plt.ylim(0,0.3)

plt.xlabel('# Epochs')

plt.legend()

plt.ylabel("Loss - Binary Cross Entropy")

plt.title('Zoom Near Zero - Loss Evolution')



plt.subplot(143)

plt.plot(history.history['acc'], label='training')

plt.plot(history.history['val_acc'], label='validation')

plt.xlabel('# Epochs')

plt.ylabel("Accuracy")

plt.legend()

plt.title('Accuracy Evolution')



plt.subplot(144)

plt.plot(history.history['acc'], label='training')

plt.plot(history.history['val_acc'], label='validation')

plt.ylim(0.9,1)

plt.xlabel('# Epochs')

plt.ylabel("Accuracy")

plt.legend()

plt.title('Zoom Near One - Accuracy Evolution')
#build the model, same as above

cactus_model_aug = Sequential()

cactus_model_aug.add(Conv2D(12, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 3))) #activation layer



#additional learning layers

cactus_model_aug.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

cactus_model_aug.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

cactus_model_aug.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

cactus_model_aug.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))



#final prediction layers

cactus_model_aug.add(Flatten())

cactus_model_aug.add(Dense(100, activation='relu'))

cactus_model_aug.add(Dense(num_classes, activation='softmax'))



#compile the model

cactus_model_aug.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True)  # randomly flip images



datagen.fit(train_data)
cactus_model_aug.fit_generator(datagen.flow(train_data,train_labels),

                              epochs = 15, validation_data = (val_data,val_labels), steps_per_epoch=20)
#aug was NOT better
test_dir = '../input/test/test'

test_paths = [join(test_dir,filename) for filename in os.listdir(test_dir)]

test_paths[0:5]
len(os.listdir(test_dir))
from IPython.display import Image, display

for i, img_path in enumerate(test_paths[0:5]):

    display(Image(img_path))
test_data = read_and_prep_images(test_paths)
np.shape(test_data)
#Get predictions

preds_test = cactus_model.predict(test_data)



# #the model returns a list of probabilities for each outcome. 

realPreds = preds_test[:,0]

realPreds[0:12]
# Save test predictions to file

# no aug performed better

output = pd.DataFrame({'id': os.listdir(test_dir),

                       'has_cactus': realPreds})

output.to_csv('submission.csv', index=False)