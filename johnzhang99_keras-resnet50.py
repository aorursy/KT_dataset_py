# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))

from pathlib import Path



data_root = Path('../input/data')

print('data_root:',data_root)



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import ResNet50

from tensorflow.keras import layers

from tensorflow.keras.optimizers import Adam
# constants

DIM = 256

BATCH_SIZE = 64

NUM_CLASSES = 15

# EPOCH = 15



TRAIN_STEPS_PER_EPOCH = 2392 // BATCH_SIZE

VAL_STEPS_PER_EPOCH = 593 // BATCH_SIZE

TEST_STEPS_PER_EPOCH = 1500 // BATCH_SIZE
# data generators

datagen = ImageDataGenerator(rescale=1./255,

                             validation_split=0.2,

                            )

train_gen = datagen.flow_from_directory(data_root/'train',

                                       target_size=(DIM,DIM),

                                       batch_size=BATCH_SIZE,

                                       subset='training',

                                       )

val_gen = datagen.flow_from_directory(data_root/'train',

                                      target_size=(DIM,DIM),

                                      batch_size=BATCH_SIZE,

                                      subset='validation',

                                      )
x_batch, y_batch = next(test_gen)

print(x_batch[0].shape)

plt.imshow(x_batch[0][:,:,0])

print(y_batch[10])

plt.show()

plt.imshow(x_batch[0][:,:,2])

plt.show()

# for i in range (0,32):

#     image = x_batch[i]

#     plt.imshow(image.transpose(2,1,0))

#     plt.show()
# building the model



model = tf.keras.Sequential()

resnet = ResNet50(include_top=False,

                  weights='imagenet',

                  input_shape=(DIM,DIM,3),

                  pooling='avg',

                 )

model.add(resnet)

model.add(layers.Dense(64, 

                       kernel_regularizer=tf.keras.regularizers.l2(0.001),

                       activation='sigmoid'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(64, 

                       kernel_regularizer=tf.keras.regularizers.l2(0.001),

                       activation='sigmoid'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(NUM_CLASSES, activation='softmax'))



# fix base



# model.layers[0].trainable = False



print('finished loading model.')
# model compile and summary

model.compile(optimizer=Adam(lr=3e-4), 

              loss='categorical_crossentropy', 

              metrics=['accuracy'],

             )



print(model.summary())
# training

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



cb_early_stopper = EarlyStopping(monitor='val_loss', patience=10)

cb_checkpointer = ModelCheckpoint(filepath='../working/best.hdf5', 

                                  monitor='val_loss', 

                                  save_best_only=True, 

                                  mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=5, min_lr=5e-5)



fit_history = model.fit_generator(

        train_gen,

        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,

        epochs=100,

        validation_data=val_gen,

        validation_steps=VAL_STEPS_PER_EPOCH,

        callbacks=[cb_checkpointer, cb_early_stopper, reduce_lr]

)

# training stats



def plot_history(histories, key='acc'):

  plt.figure(figsize=(16,10))



  for name, history in histories:

    val = plt.plot(history.epoch, history.history['val_'+key],

                   '--', label=name.title()+' Val')

    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),

             label=name.title()+' Train')



  plt.xlabel('Epochs')

  plt.ylabel(key.replace('_',' ').title())

  plt.legend()



  plt.xlim([0,max(history.epoch)])



# plot history

plot_history([('model', fit_history)])
# inference



model.load_weights("../working/best.hdf5")

print('model weights loaded')



test_gen = datagen.flow_from_directory(data_root/'test',

                                      target_size=(DIM,DIM),

                                      batch_size=BATCH_SIZE,

                                      shuffle=False

                                      )

test_gen.reset()



pred = model.predict_generator(test_gen, steps = TEST_STEPS_PER_EPOCH, verbose = 1)

predicted_class_indices = np.argmax(pred, axis = 1)

# output prediction result



filenames = []

for i in range(TEST_STEPS_PER_EPOCH):

    idx = (test_gen.batch_index - 1) * test_gen.batch_size

    filenames += test_gen.filenames[idx : idx + test_gen.batch_size]

    

filenames_id = [Path(fname).stem for fname in filenames]

# print(filenames_id)



# print(predicted_class_indices[5])

# print(filenames_id[5])



back2label = {}

for k,v in train_gen.class_indices.items():

    back2label[v] = k

print(back2label)



result = []

for i in range(len(filenames_id)):

    result.append((filenames_id[i],back2label[predicted_class_indices[i]]))

print(result[:5])



import csv

with open('../working/submission.csv', 'w') as csv_file:

    writer = csv.writer(csv_file, delimiter=',')

    writer.writerow(['Id', 'Category'])

    for (pid, class_name) in result:

        writer.writerow([pid,class_name])



print('done!')