def prepare_tf():

    !pip uninstall tensorflow -y

    !pip install tensorflow-gpu==2.0.0-alpha0

    from tensorflow.python.ops import control_flow_util

    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

    from tensorflow.python.client import device_lib 

    print(device_lib.list_local_devices())
import tensorflow as tf

print(tf.__version__)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

print(os.listdir("../input/train/"))

print(len(os.listdir("../input/train/train/")))

sample_submission_file = pd.read_csv("../input/sample_submission_file.csv")

print(len(os.listdir("../input/extraimages/extraimages/")))
df_sample_submission_file.head()
def append_ext(fn):

    return fn+".jpg"



testdf=pd.read_csv("../input/sample_submission_file.csv",dtype=str)

testdf["Id"]=testdf["Id"].apply(append_ext)
CLASS_MODE = 'categorical'
from pathlib import Path

from PIL import Image



def read_pil_image(img_path, height, width):

        with open(img_path, 'rb') as f:

            return np.array(Image.open(f).convert('RGB').resize((width, height)))



def load_all_images(dataset_path, height, width, img_ext='jpg'):

    return np.array([read_pil_image(str(p), height, width) for p in 

                                    Path(dataset_path).rglob("*."+img_ext)]) 



IMAGE_HT_WID=96

BATCH_SIZE = 32 #100 



from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop

train_datagen = ImageDataGenerator(

                               rotation_range=8,

                               width_shift_range=0.1,

                               height_shift_range=0.1,

                               shear_range=0.01,

                               zoom_range=[0.9, 1.25],

                               horizontal_flip=True,

                               vertical_flip=False,

                              channel_shift_range = 0.1,

                              fill_mode='nearest',

                              #brightness_range=[0.5, 1.5],

                               validation_split=0.25,

                               rescale=1./255)



train_generator=train_datagen.flow_from_directory(

                    directory="../input/train/train/",

                    subset="training",

                   batch_size=BATCH_SIZE,

                    seed=42,

                    shuffle=True,

                    class_mode='categorical',

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))



valid_generator=train_datagen.flow_from_directory(

                    directory="../input/train/train/",

                    subset="validation",

                    batch_size=BATCH_SIZE,

                    seed=42,

                    shuffle=True,

                    class_mode='categorical',

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(

dataframe=testdf,

directory="../input/test/test",

x_col="Id",

y_col=None,

batch_size=32,

seed=42,

shuffle=False,

class_mode=None,

target_size=(IMAGE_HT_WID,IMAGE_HT_WID))
import tensorflow as tf

keras = tf.keras
np.random.seed(42)

import random as rn

rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,

                           inter_op_parallelism_threads=1)



from keras import backend as K



tf.set_random_seed(1234)



sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)

#https://www.tensorflow.org/tutorials/images/transfer_learning



from tensorflow import keras

import tensorflow as tf

keras = tf.keras

base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_HT_WID,IMAGE_HT_WID, 3),

                                               include_top=False, 

                                               weights='imagenet')

base_model.trainable = False

print(base_model.summary())

model = tf.keras.Sequential([

  base_model,

  keras.layers.GaussianNoise(0.2),  

  keras.layers.GlobalAveragePooling2D(),

  keras.layers.Dense(5, activation='sigmoid')

])



model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), metrics=['accuracy'])



print(model.summary())

len(base_model.layers)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, 

                                                                    patience=3, verbose=2, mode='auto',

                                                                    min_lr=1e-6)



early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 

                                            # min_delta=0, 

                                             patience=6, verbose=2, mode='auto',

                                             baseline=None, restore_best_weights=True)
from keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras import regularizers, optimizers

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=(IMAGE_HT_WID,IMAGE_HT_WID,3)))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(output_dim=5, activation='sigmoid'))

model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=10

                   )
model.evaluate_generator(generator=valid_generator,

steps=STEP_SIZE_VALID)
test_generator.reset()

pred=model.predict_generator(test_generator,

steps=STEP_SIZE_TEST,

verbose=1)
batch_size=32

nb_validation_samples=1412

steps = nb_validation_samples / batch_size

predicted_class_indices = model.predict_generator(valid_generator, steps)
labels = (train_generator.class_indices)

labels
labels = dict((v,k) for k,v in labels.items())

labels
predicted_class_indices[:10]

import numpy as np

predicted_class_indices=hash((np.array(predicted_class_indices[:])))

predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames

results=pd.DataFrame({"Category":predictions,

                      "id":filenames})

#subm = pd.merge(df_test, results, on='file_name')[['id','Category']]

results.head()
results.Category.value_counts()
results.loc[:,'id'] = results.id.str.replace('0/','')

results.head()
results.to_csv("submission.csv",index=False)