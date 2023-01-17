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
# import tensorflow as tf



# print("TensorFlow Version:",tf.__version__)



# # Detect hardware, return appropriate distribution strategy

# try:

#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

#     print('Running on TPU ', tpu.master())

# except ValueError:

#     tpu = None

    

# if tpu:

#     tf.config.experimental_connect_to_cluster(tpu)

#     tf.tpu.experimental.initialize_tpu_system(tpu)

#     strategy = tf.distribute.experimental.TPUStrategy(tpu)

# else:

#     strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



# print("REPLICAS: ", strategy.num_replicas_in_sync)

from zipfile import ZipFile, is_zipfile



source = "../input/dogs-vs-cats/"

for file_ in os.listdir(source):

    file_path = os.path.join(source, file_)

    if is_zipfile(file_path):

        with ZipFile(file_path, 'r') as zipref:

            zipref.extractall()
# import shutil



# shutil.rmtree('./train/', ignore_errors=True)

# shutil.rmtree('./test1/', ignore_errors=True)
!mkdir './train/cat'

!mkdir './train/dog'

!bash -c 'mv ./train/cat.{0..12499}.jpg ./train/cat/'

!bash -c 'mv ./train/dog.{0..12499}.jpg ./train/dog/'

print("Train dataset contains %i folders: %i cats and %i dogs" %(len(os.listdir('./train/')),

                                                                 len(os.listdir('./train/cat')), 

                                                                 len(os.listdir('./train/dog'))))

print("Test dataset contains %i samples" %(len(os.listdir('./test1/'))))
from keras.preprocessing.image import ImageDataGenerator



train_dir = './train'

test_dir = './test1'



train_cat_dir = train_dir + '/cat'

train_dog_dir = train_dir + '/dog'



train_data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# add validation split was added train dataset into 'training' and 'validation' subsets



test_data_generator = ImageDataGenerator(rescale=1./255)



BATCH_SIZE = 20



args_ = {"target_size":(150, 150,),

         "batch_size":BATCH_SIZE,

         "class_mode":'binary'}

train_generator = train_data_generator.flow_from_directory(

                    train_dir,

                    subset='training',

                    **args_)



validation_generator = train_data_generator.flow_from_directory(

                    train_dir,

                    subset='validation',

                    **args_)

test_generator = test_data_generator.flow_from_directory(

                    test_dir,

                    **args_)
print('data batch shape:', train_generator[0][0].shape)

print('labels batch shape:', train_generator[0][1].shape)
from keras import models, layers, optimizers



def build_model():

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))



    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))



    model.compile(loss='binary_crossentropy',

                 optimizer=optimizers.RMSprop(lr=1e-4),

                 metrics=['acc',])



    return model
model = build_model()

model.summary()
history = model.fit_generator(train_generator,

                         steps_per_epoch=train_generator.samples // BATCH_SIZE, 

                         epochs=25,

                         validation_data=validation_generator,

                         validation_steps=validation_generator.samples // BATCH_SIZE)
model.save('cats and dogs_v1.h5')
# from IPython.display import FileLink

# FileLink('cats and dogs_v1.h5')
import plotly.graph_objects as go

fig1 = go.Figure()

fig2 = go.Figure()



acc = history.history.get('acc')

val_acc = history.history.get('val_acc')

loss = history.history.get('loss')

val_loss = history.history.get('val_loss')



epochs = list(range(1, len(acc) + 1))



fig1.add_scatter(x=epochs, y=acc, name='Training acc', mode='markers')

fig1.add_scatter(x=epochs, y=val_acc, name="Validation acc")

fig1.update_layout(title={"text":"Training and validation Accuracy", "x":0.5}, xaxis_title="Epoch", yaxis_title="Accuracy")

fig1.show()



fig2.add_scatter(x=epochs, y=loss, name='Training loss', mode='markers')

fig2.add_scatter(x=epochs, y=val_loss, name="Validation loss")

fig2.update_layout(title={"text":"Training and validation Loss", "x":0.5}, xaxis_title="Epoch", yaxis_title="loss")

fig2.show()
