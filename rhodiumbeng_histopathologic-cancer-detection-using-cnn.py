import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
# load data from csv files

train_df = pd.read_csv('../input/train_labels.csv')

test_df = pd.read_csv('../input/sample_submission.csv')

print(train_df.shape, test_df.shape)
train_df['id'] = train_df['id'].apply(lambda x: x+'.tif')

test_df['id'] = test_df['id'].apply(lambda x: x+'.tif')
train_df['label'] = train_df['label'].astype(str)
train_df['label'].value_counts()
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array



train_path = '../input/train/'

test_path = '../input/test/'
# look at some of the pics from train_df labelled '1'

positive = train_df[train_df['label']=='1']

plt.figure(figsize=(15,7))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(load_img(train_path+positive.iloc[i]['id']))

    plt.title("label=%s" % positive.iloc[i]['label'], y=1)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# look at some of the pics from train_df labelled '0'

negative = train_df[train_df['label']=='0']

plt.figure(figsize=(15,7))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(load_img(train_path+negative.iloc[i]['id']))

    plt.title("label=%s" % negative.iloc[i]['label'], y=1)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
from keras_preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)
# set up two data generators; (1) training, (2) validation from train set

n_x = 96

train_generator = datagen.flow_from_dataframe(dataframe=train_df, 

                                              directory=train_path, 

                                              target_size=(n_x,n_x), 

                                              x_col='id', y_col='label', 

                                              subset='training', 

                                              batch_size=128, seed=12, 

                                              class_mode='categorical')
valid_generator = datagen.flow_from_dataframe(dataframe=train_df, 

                                              directory=train_path,

                                              target_size=(n_x,n_x), 

                                              x_col='id', y_col='label', 

                                              subset='validation', 

                                              batch_size=128, seed=12, 

                                              class_mode='categorical')
# set up data generator for test set

test_datagen = ImageDataGenerator(rescale=1./255.)

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df, 

                                                  directory=test_path, 

                                                  target_size=(n_x,n_x), 

                                                  x_col='id', y_col=None, 

                                                  batch_size=1, seed=12, 

                                                  shuffle=False, 

                                                  class_mode=None)
# define step sizes for model training

step_size_train = train_generator.n//train_generator.batch_size

step_size_valid = valid_generator.n//valid_generator.batch_size

step_size_test = test_generator.n//test_generator.batch_size

print(step_size_train, step_size_valid, step_size_test)
# build the CNN from keras

from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=5, activation='relu', input_shape=(96, 96, 3)))

model.add(layers.Conv2D(32, kernel_size=5, activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Conv2D(64, kernel_size=5, activation='relu'))

model.add(layers.Conv2D(64, kernel_size=5, activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Conv2D(128, kernel_size=5, activation='relu'))

model.add(layers.Conv2D(128, kernel_size=5, activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(2, activation='softmax'))



model.summary()
# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', 

              metrics=['accuracy'])
# Train and validate the model

epochs = 20

history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=step_size_train, 

                              validation_data=valid_generator, 

                              validation_steps=step_size_valid,

                              epochs=epochs)
# plot and visualise the training and validation losses

loss = history.history['loss']

dev_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)



from matplotlib import pyplot as plt

plt.figure(figsize=(15,10))

plt.plot(epochs, loss, 'bo', label='training loss')

plt.plot(epochs, dev_loss, 'b', label='validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# predict on test set

test_generator.reset()

pred = model.predict_generator(test_generator, steps=step_size_test, 

                               verbose=1)
# create submission file

sub = pd.read_csv('../input/sample_submission.csv')

sub['label'] = pred[:,0]

sub.head()
# generate submission file in csv format

sub.to_csv('submission.csv', index=False)