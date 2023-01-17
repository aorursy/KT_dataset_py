# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split



import keras

from keras.models import Model



from keras.layers import GlobalMaxPooling2D

from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input

from keras.layers.convolutional import Conv2D, MaxPooling2D



from keras.applications.mobilenet import MobileNet

from keras.applications.inception_v3 import InceptionV3



from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K



import os



# Hyperparams

IMAGE_SIZE = 96

IMAGE_CHANNELS = 3



SAMPLE_SIZE = 88800



validation_split=0.1



train_df = pd.read_csv("../input/train_labels.csv")

# Remove error image

train_df = train_df[train_df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']

# Remove error black image

train_df = train_df[train_df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']



train_df['filename'] = train_df['id'] + ".tif"

train_df['class'] = train_df['label']

train_df['class'] = train_df['class'].apply(str)



# test_df = pd.DataFrame({'filename':os.listdir(test_data_dir)})



nb_train_samples = train_df.shape[0] - train_df.shape[0]*validation_split

nb_validation_samples = nb_train_samples*validation_split



if K.image_data_format() == 'channels_first':

    input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)

else:

    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    

train_data_dir = '../input/train'

test_data_dir = '../input/test'
train_batch_size = 64

val_batch_size = 32



train_datagen = ImageDataGenerator(rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    channel_shift_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    rotation_range=45,

    fill_mode='reflect',

    vertical_flip=True,

    validation_split=validation_split)



train_generator = train_datagen.flow_from_dataframe(

    dataframe = train_df,

    directory = train_data_dir,

    target_size = (IMAGE_SIZE, IMAGE_SIZE),

    shuffle=True,

    batch_size=train_batch_size,

    subset="training",

    class_mode = 'binary')



valid_generator = train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory=train_data_dir,

    target_size = (IMAGE_SIZE, IMAGE_SIZE),

    shuffle=False,

    batch_size=val_batch_size,

    subset="validation",

    class_mode = 'binary')





train_steps = np.ceil(train_generator.samples // train_batch_size)

val_steps = np.ceil(valid_generator.samples // val_batch_size)
from keras.layers import GaussianNoise

from keras.layers import Concatenate, GlobalMaxPooling2D, Activation



input_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

input = Input(shape=input_shape)

# corr_input = GaussianNoise(1.0)(input)



base_model = MobileNet(input_tensor=input, include_top=False, weights="imagenet")

# base_model = InceptionV3(input_tensor=input, weights="imagenet", include_top=False)



out = Dropout(0.5)(base_model.output)

out = Conv2D(1, kernel_size=(3, 3), padding="valid", activation="sigmoid")(out)

out = Flatten()(out)

# out = Activation('sigmoid')(out)



model = Model(inputs=base_model.inputs, outputs=out)



# serialize model to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

    

    

print("Done")
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])

# model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.summary()
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1, mode='max', min_lr=0.00000001)

checkpoint = ModelCheckpoint("model_best.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit_generator(train_generator, steps_per_epoch=train_steps, validation_data=valid_generator, 

                              validation_steps=val_steps, epochs=15, verbose=1, callbacks=[reduce_lr, checkpoint])
model.load_weights("model_best.h5")
df_test = pd.DataFrame({'id':os.listdir(test_data_dir)})

df_test['id'] = [f.split(sep='.')[0] for f in df_test['id']]

df_test['filename'] = df_test['id'] + ".tif"



test_batch_size = 2



test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(

    dataframe=df_test,

    directory=test_data_dir,

    target_size = (IMAGE_SIZE, IMAGE_SIZE),

    shuffle=False,

    batch_size=test_batch_size,

    class_mode = None)
test_steps = test_generator.samples / test_batch_size

test_generator.reset()

pred = model.predict_generator(test_generator, steps=test_steps, verbose=1)
df_preds = pd.DataFrame(pred, columns=['label'])



test_filenames = test_generator.filenames

df_preds['file_names'] = test_filenames



def extract_id(x):

    # split into a list

    a = x.split('/')

    # split into a list

    b = a[0].split('.')

    extracted_id = b[0]



    return extracted_id





df_preds['id'] = df_preds['file_names'].apply(extract_id)

y_pred = df_preds['label']

image_id = df_preds['id']



submission = pd.DataFrame({'id': image_id, 'label': y_pred, }).set_index('id')

submission.to_csv('preds.csv', columns=['label'])



print("Done")



df_preds.head()