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
path = '../input/Kannada-MNIST/'
import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image
from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical



from keras.preprocessing.image import ImageDataGenerator



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D



from keras.callbacks import EarlyStopping
"""

helper function to show a number of randomly selected images 

belonging either to a specified label or selected across all labels

"""



def show_random_images(images, num=10, label=None):



    # generating images' subsample if label specified

    if label is not None:

        images = images[images.label == label]

    

    fig, axs = plt.subplots(num, figsize=(1.25, num * 2.5))

    

    for i in range(num):

    

        rnd = np.random.randint(len(images))

    

        # getting image data and splitting between label and pixels' vector

        img_data = np.array(images.iloc[rnd], dtype='uint8')    

        img_label = img_data[0]

        img_pixels = img_data[1:]

        

        # reshaping image to 2D array

        img_shape = (int(np.sqrt(img_pixels.shape[0])), int(np.sqrt(img_pixels.shape[0])))

        img_array = img_pixels.reshape(img_shape)

        

        title = 'Image {} / labelled {}'.format(rnd, img_label)

        

        axs[i].imshow(img_array, alpha=0.66, cmap='gray')

        axs[i].set_title(title)
train_data = pd.read_csv(path + 'train.csv')

train_data
# checking labels distribution



train_data.label.value_counts()
show_random_images(train_data, num=5, label=5)
dig_data = pd.read_csv(path + 'Dig-MNIST.csv')

dig_data
# checking labels distribution



dig_data.label.value_counts()
show_random_images(dig_data, num=5, label=5)
# helper function to show randomly selected image from 2D images array



def show_random_image(imgset):

    

    rnd = np.random.randint(imgset.shape[0])

    imgarray = imgset[rnd,:,:,0]

    plt.figure(figsize=(1.5, 1.5))

    plt.imshow(imgarray, cmap='gray')
# preparing train image labels using 'one-hot' encoding



train_labels = to_categorical(train_data.label)

train_labels
train_labels.shape
# preparing train images array ('flat' image vectors)



train_images = np.array(train_data.drop(columns='label'))

train_images.shape
# preparing 2D train images array (reshaping original 'flat' image vectors array)



n_images = train_images.shape[0]

dim = int(np.sqrt(train_images.shape[1]))



train_images_2D = train_images.reshape(n_images, dim, dim, 1)

train_images_2D.shape
show_random_image(train_images_2D)
# normalizing "train" images



train_images_2D = train_images_2D.astype('float')

train_images_2D /= 255
# preparing dig-mnist image labels using 'one-hot' encoding



dig_labels = to_categorical(dig_data.label)

dig_labels
dig_labels.shape
# preparing train images array ('flat' image vectors)



dig_images = np.array(dig_data.drop(columns='label'))

dig_images.shape
# preparing 2D dig-mnist images array (reshaping original 'flat' image vectors array)



n_images = dig_images.shape[0]

dim = int(np.sqrt(dig_images.shape[1]))



dig_images_2D = dig_images.reshape(n_images, dim, dim, 1)

dig_images_2D.shape
show_random_image(dig_images_2D)
# normalizing "Dig-MNIST" images



dig_images_2D = dig_images_2D.astype('float')

dig_images_2D /= 255
images_train,imames_val,labels_train,labels_val = train_test_split(train_images_2D, train_labels, 

                                                                   random_state=42,test_size=0.15)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(images_train)
test_data = pd.read_csv(path + 'test.csv', index_col='id')

test_data
submission = pd.read_csv(path + 'sample_submission.csv', index_col='id')

submission
# preparing test images array ('flat' image vectors)



test_images = np.array(test_data)

test_images.shape
# preparing 2D test images array (reshaping original 'flat' image vectors array)



n_images = test_images.shape[0]

dim = int(np.sqrt(train_images.shape[1]))



test_images_2D = test_images.reshape(n_images, dim, dim, 1)

test_images_2D.shape
# normalizing "test" images



test_images_2D = test_images_2D.astype('float')

test_images_2D /= 255
# setting input dimensionality - 2D image arrays

input_shape = (dim, dim, 1)

num_classes = 10
# setting optimization parameters

optimizer = 'rmsprop'

loss = 'categorical_crossentropy'

metrics = ['accuracy']
# setting training parameters

epochs = 100

batch_size = 1024



early_stop = EarlyStopping(monitor='val_loss', 

                           min_delta=0, 

                           patience=3, 

                           verbose=True, 

                           mode='auto', 

                           baseline=None, 

                           restore_best_weights=False)



callbacks = [early_stop]
# defining model's architecture - version 1

# source: https://keras.io/examples/mnist_cnn/



model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_classes, activation='softmax'))
# defining model's architecture - version 2



kernel_size = (5, 5)



model = Sequential()



model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', 

                 input_shape=input_shape))

model.add(Conv2D(64, kernel_size=kernel_size, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_classes, activation='softmax'))
# defining model's architecture - version 3



kernel_size = (5, 5)



model = Sequential()



model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', 

                 input_shape=input_shape))

model.add(Conv2D(32, kernel_size=kernel_size, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size=kernel_size, activation='relu'))

model.add(Conv2D(64, kernel_size=kernel_size, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_classes, activation='softmax'))
# defining model's architecture - version 4



kernel_size_1 = (7, 7)

kernel_size_2 = (5, 5)



model = Sequential()



model.add(Conv2D(32, kernel_size=kernel_size_1, activation='relu', padding='same', 

                 input_shape=input_shape))

model.add(Conv2D(32, kernel_size=kernel_size_1, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size=kernel_size_2, activation='relu', padding='same'))

model.add(Conv2D(64, kernel_size=kernel_size_2, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_classes, activation='softmax'))
# defining model's architecture - version 5



kernel_size_1 = (7, 7)

kernel_size_2 = (5, 5)



model = Sequential()



model.add(Conv2D(32, kernel_size=kernel_size_1, activation='relu', padding='same', 

                 input_shape=input_shape))

model.add(Conv2D(32, kernel_size=kernel_size_1, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size=kernel_size_2, activation='relu', padding='same'))

model.add(Conv2D(64, kernel_size=kernel_size_2, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_classes, activation='softmax'))
# defining model's architecture - version 7



kernel_size_1 = (7, 7)

kernel_size_2 = (5, 5)



model = Sequential()



model.add(Conv2D(32, kernel_size=kernel_size_1, activation='relu', padding='same', 

                 input_shape=input_shape))

model.add(Conv2D(32, kernel_size=kernel_size_1, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size=kernel_size_2, activation='relu', padding='same'))

model.add(Conv2D(64, kernel_size=kernel_size_2, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_classes, activation='softmax'))
# defining model's architecture - version 8



kernel_size_1 = (7, 7)

kernel_size_2 = (5, 5)



model = Sequential()



model.add(Conv2D(32, kernel_size=kernel_size_1, activation='relu', padding='same', 

                 input_shape=input_shape))

model.add(Conv2D(32, kernel_size=kernel_size_1, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size=kernel_size_2, activation='relu', padding='same'))

model.add(Conv2D(64, kernel_size=kernel_size_2, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()
model.fit_generator(datagen.flow(images_train, labels_train, 

                                 batch_size=batch_size), 

                    epochs=epochs, 

                    verbose=True, 

                    callbacks=callbacks, 

                    validation_data=(imames_val, labels_val))
# showing history of 'accuracy'



plt.figure()

plt.plot(model.history.history['accuracy'], label='TRAIN ACC')

plt.plot(model.history.history['val_accuracy'], label='VAL ACC')

plt.legend()

plt.show()
# showing history of 'loss'



plt.figure()

plt.plot(model.history.history['loss'], label='TRAIN LOSS')

plt.plot(model.history.history['val_loss'], label='VAL LOSS')

plt.legend()

plt.show()
# making predictions for "train" data (in-sample check)



pred_train = model.predict_classes(train_images_2D)

pred_train.shape
hits = (pred_train == train_data.label)

print('Hits: {}, i.e. {:.2f}%'.format(hits.sum(), hits.sum() / pred_train.shape[0] * 100))
miss = (pred_train != train_data.label)

print('Misses: {}, i.e. {:.2f}%'.format(miss.sum(), miss.sum() / pred_train.shape[0] * 100))
cm = confusion_matrix(y_true=train_data.label, y_pred=pred_train)

cm = pd.DataFrame(cm, index=range(num_classes), columns=range(num_classes))

cm
plt.figure(figsize=(12,10))

sns.heatmap(cm, annot=True,)
# evaluating model on "train" data



eval_metrics = model.evaluate(x=train_images_2D, y=train_labels, 

                              batch_size=batch_size, verbose=True, callbacks=callbacks)

pd.DataFrame(eval_metrics, index=model.metrics_names, columns=['metric'])
# making predictions for "Dig-MNIST" data



pred_Dig = model.predict_classes(dig_images_2D)

pred_Dig.shape
hits = (pred_Dig == dig_data.label)

print('Hits: {}, i.e. {:.2f}%'.format(hits.sum(), hits.sum() / pred_Dig.shape[0] * 100))
miss = (pred_Dig != dig_data.label)

print('Misses: {}, i.e. {:.2f}%'.format(miss.sum(), miss.sum() / pred_Dig.shape[0] * 100))
cm = confusion_matrix(y_true=dig_data.label, y_pred=pred_Dig)

cm = pd.DataFrame(cm, index=range(num_classes), columns=range(num_classes))

cm
plt.figure(figsize=(12,10))

sns.heatmap(cm, annot=True,)
# evaluating model on "Dig-MNIST" data



eval_metrics = model.evaluate(x=dig_images_2D, y=dig_labels, 

                              batch_size=batch_size, verbose=True, callbacks=callbacks)

pd.DataFrame(eval_metrics, index=model.metrics_names, columns=['metric'])
# setting the optimal number of epochs

epochs = early_stop.stopped_epoch + 1



# re-training the model on full train dataset

model.fit(train_images_2D, train_labels, 

          batch_size=batch_size, epochs=epochs, 

          verbose=True)
# making predictions on "test" data



pred_test = model.predict_classes(test_images_2D)
submission.label = pred_test

submission.to_csv('submission.csv')