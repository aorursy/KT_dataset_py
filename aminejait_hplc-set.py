import h5py

import pandas as pd

import numpy as np

import os

import math

from glob import glob

from sklearn.utils import shuffle

import shutil

from sklearn.model_selection import train_test_split

from keras.initializers import glorot_uniform

from keras.models import Model, load_model, Sequential

from keras import optimizers

from keras import regularizers

from keras.applications.resnet50 import ResNet50

from keras.applications.inception_v3 import InceptionV3

from keras.layers import Input, Add, Dense, Activation,GlobalAveragePooling2D, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout

from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau

from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

from PIL import Image

from IPython.display import clear_output

df = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')

#df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']

#df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']

df_0 = df[df['label'] == 0].sample(89000, random_state = 101)

df_1 = df[df['label'] == 1].sample(89000, random_state = 101)

df = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)

df = shuffle(df)

df['id'] =df.id.map(lambda x:x +'.tif')

df['label'] = df['label'].astype(str)

df_sample = df.sample(n=10000, random_state=2018)
test_df = pd.DataFrame({'filename':os.listdir('../input/histopathologic-cancer-detection/test')})



clear_output()

image_gen = ImageDataGenerator( samplewise_center = True, brightness_range = (0.1,0.2),zca_whitening = True,

                validation_split=0.2, vertical_flip = True, horizontal_flip = True)

image_gen_test = ImageDataGenerator( vertical_flip = True, horizontal_flip = True, brightness_range = (0.1,0.2),

                  samplewise_center = True, zca_whitening = True)




clear_output()

img_iter = image_gen.flow_from_dataframe(

    df_sample,

    shuffle=True,

    directory= '../input/histopathologic-cancer-detection/train',

    x_col='id',

    y_col='label',

    class_mode='binary',

    color_mode = 'rgb',

    target_size=(96, 96),

    batch_size=256,

    subset='training'

)



img_iter_val = image_gen.flow_from_dataframe(

    df_sample,

    shuffle=False,

    directory= '../input/histopathologic-cancer-detection/train',

    x_col='id',

    y_col='label',

    class_mode='binary',

    color_mode = 'rgb',

    target_size=(96, 96),

    batch_size=256,

    subset='validation'

)





#test_generator = image_gen_test.flow_from_dataframe(

        #dataframe = test_df,

        #directory = '../input/histopathologic-cancer-detection/test',

        #target_size=(96, 96),

        #color_mode = 'rgb',

        #shuffle = False,

        #class_mode=None,

        #batch_size=256)


base_model = InceptionV3(weights = None, input_shape = (96,96,3), include_top=False)

# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes

x = Dense(200, activation='relu')(x)

predictions = Dense(1, activation = 'sigmoid')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)
clear_output()

#model = my_model

adam = optimizers.Adam(lr=0.00007)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])


clear_output()

STEP_SIZE_TRAIN=np.ceil(img_iter.n//img_iter.batch_size)+1

STEP_SIZE_VALID=np.ceil(img_iter_val.n//img_iter_val.batch_size)+1

#STEP_SIZE_TEST=np.ceil(test_generator.n//test_generator.batch_size)+1



history = model.fit_generator(img_iter, steps_per_epoch=STEP_SIZE_TRAIN,

                    epochs= 13, validation_data = img_iter_val, validation_steps = STEP_SIZE_VALID)
def show_final_history(history):

    fig, ax = plt.subplots(1, 2, figsize=(15,5))

    ax[0].set_title('loss')

    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1].set_title('acc')

    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")

    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")

    ax[0].legend()

    ax[1].legend()
show_final_history(history)
model_json = model.to_json()

model.save("model.h5")