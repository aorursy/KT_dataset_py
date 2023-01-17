import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import tensorflow as tf

from tensorflow import keras



import os

from shutil import copyfile, move

from tqdm import tqdm

import h5py
print(tf.__version__)

print(tf.test.is_gpu_available())
training_df = pd.read_csv("../input/train.csv")

training_df.head()
src = "../input/train/train/"

dst = "../sorted_training/"



os.mkdir(dst)

os.mkdir(dst+"true")

os.mkdir(dst+"false")



with tqdm(total=len(list(training_df.iterrows()))) as pbar:

    for idx, row in training_df.iterrows():

        pbar.update(1)

        if row["has_cactus"] == 1:

            copyfile(src+row["id"], dst+"true/"+row["id"])

        else:

            copyfile(src+row["id"], dst+"false/"+row["id"])
src = "../sorted_training/"

dst = "../sorted_validation/"



os.mkdir(dst)

os.mkdir(dst+"true")

os.mkdir(dst+"false")



validation_df = training_df.sample(n=int(len(training_df)/10))



with tqdm(total=len(list(validation_df.iterrows()))) as pbar:

    for idx, row in validation_df.iterrows():

        pbar.update(1)

        if row["has_cactus"] == 1:

            move(src+"true/"+row["id"], dst+"true/"+row["id"])

        else:

            move(src+"false/"+row["id"], dst+"false/"+row["id"])
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import InputLayer, Input

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation

from tensorflow.keras.layers import BatchNormalization, Reshape, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
batch_size = 64



train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    horizontal_flip=True,

    vertical_flip=True)



train_data_dir = "../sorted_training"

train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    shuffle=True,

    target_size=(32, 32),

    batch_size=batch_size,

    class_mode='binary')





validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_data_dir = "../sorted_validation"

validation_generator = validation_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(32, 32),

    batch_size=batch_size,

    class_mode='binary')



input_shape = (32,32,3)

num_classes = 2

dropout_dense_layer = 0.6



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))



model.add(Flatten())

model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dropout(dropout_dense_layer))



model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(dropout_dense_layer))



model.add(Dense(1))

model.add(Activation('sigmoid'))
model.compile(loss=keras.losses.binary_crossentropy,

              optimizer=keras.optimizers.Adam(lr=0.001),

              metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=25),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
epochs = 100

history = model.fit_generator(train_generator,

          validation_data=validation_generator,

          epochs=epochs,

          verbose=1,

          shuffle=True,

          callbacks=callbacks)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.show()
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.show()
model.load_weights("best_model.h5")
test_folder = "../input/test/"

test_datagen = ImageDataGenerator(

    rescale=1. / 255)



test_generator = test_datagen.flow_from_directory(

    directory=test_folder,

    target_size=(32,32),

    batch_size=1,

    class_mode='binary',

    shuffle=False

)
pred=model.predict_generator(test_generator,verbose=1)

pred_binary = [0 if value<0.50 else 1 for value in pred]  
csv_file = open("sample_submission_cnn.csv","w")

csv_file.write("id,has_cactus\n")

for filename, prediction in zip(test_generator.filenames,pred_binary):

    name = filename.split("/")[1].replace(".tif","")

    csv_file.write(str(name)+","+str(prediction)+"\n")

csv_file.close()