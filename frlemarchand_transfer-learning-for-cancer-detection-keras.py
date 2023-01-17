import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import tensorflow as tf

from tensorflow import keras



import os

from shutil import copyfile, move

from tqdm import tqdm

import h5py

import random



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation

from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.applications import VGG16
dataset_df = pd.read_csv("../input/train_labels.csv")

dataset_df["filename"] = [item.id+".tif" for idx, item in dataset_df.iterrows()]

dataset_df["groundtruth"] = ["cancerous" if item.label==1 else "healthy" for idx, item in dataset_df.iterrows()]

dataset_df.head()
training_sample_percentage = 0.8

training_sample_size = int(len(dataset_df)*training_sample_percentage)

validation_sample_size = len(dataset_df)-training_sample_size



training_df = dataset_df.sample(n=training_sample_size)

validation_df = dataset_df[~dataset_df.index.isin(training_df.index)]
training_batch_size = 64

validation_batch_size = 64

target_size = (96,96)



train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    horizontal_flip=True,

    vertical_flip=True,

    zoom_range=0.2, 

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    dataframe = training_df,

    x_col='filename',

    y_col='groundtruth',

    directory='../input/train/',

    target_size=target_size,

    batch_size=training_batch_size,

    shuffle=True,

    class_mode='binary')





validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_dataframe(

    dataframe = validation_df,

    x_col='filename',

    y_col='groundtruth',

    directory='../input/train/',

    target_size=target_size,

    shuffle=False,

    batch_size=validation_batch_size,

    class_mode='binary')

def plot_random_samples(generator):

    generator_size = len(generator)

    index=random.randint(0,generator_size-1)

    image,label = generator.__getitem__(index)



    sample_number = 10

    fig = plt.figure(figsize = (20,sample_number))

    for i in range(0,sample_number):

        ax = fig.add_subplot(2, 5, i+1)

        ax.imshow(image[i])

        if label[i]==0:

            ax.set_title("Cancerous cells")

        elif label[i]==1:

            ax.set_title("Healthy cells")

    plt.tight_layout()

    plt.show()
plot_random_samples(validation_generator)
input_shape = (96, 96, 3)

pretrained_layers = VGG16(weights='imagenet',include_top = False, input_shape=input_shape)

pretrained_layers.summary()
for layer in pretrained_layers.layers[:-8]:

    layer.trainable = False



for layer in pretrained_layers.layers:

    print(layer, layer.trainable)
dropout_dense_layer = 0.6



model = Sequential()

model.add(pretrained_layers)

    

model.add(GlobalAveragePooling2D())

model.add(Dense(256, use_bias=False))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(dropout_dense_layer))



model.add(Dense(1))

model.add(Activation('sigmoid'))
model.summary()
model.compile(loss=keras.losses.binary_crossentropy,

              optimizer=keras.optimizers.Adam(lr=0.001),

              metrics=['accuracy'])
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.5),

             EarlyStopping(monitor='val_loss', patience=5),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]



train_step_size = train_generator.n // train_generator.batch_size

validation_step_size = validation_generator.n // validation_generator.batch_size
epochs = 20

history = model.fit_generator(train_generator,

          steps_per_epoch = train_step_size,

          validation_data= validation_generator,

          validation_steps = validation_step_size,

          epochs=epochs,

          verbose=1,

          shuffle=True,

          callbacks=callbacks)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Accuracy over epochs')

plt.ylabel('Acc')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss over epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()
model.load_weights("best_model.h5")
src="../input/test"



test_folder="../test_folder"

dst = test_folder+"/test"

os.mkdir(test_folder)

os.mkdir(dst)



file_list =  os.listdir(src)

with tqdm(total=len(file_list)) as pbar:

    for filename in file_list:

        pbar.update(1)

        copyfile(src+"/"+filename,dst+"/"+filename)

        

test_datagen = ImageDataGenerator(

    rescale=1. / 255)



test_generator = test_datagen.flow_from_directory(

    directory=test_folder,

    target_size=target_size,

    batch_size=1,

    shuffle=False,

    class_mode='binary'

)
pred=model.predict_generator(test_generator,verbose=1)
csv_file = open("sample_submission.csv","w")

csv_file.write("id,label\n")

for filename, prediction in zip(test_generator.filenames,pred):

    name = filename.split("/")[1].replace(".tif","")

    csv_file.write(str(name)+","+str(prediction[0])+"\n")

csv_file.close()