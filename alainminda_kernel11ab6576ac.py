# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import json





ann_file = '../input/inaturalist-2019-fgvc6/train2019.json'

with open(ann_file) as data_file:

        train_anns = json.load(data_file)
train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id','category_id']]

train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})

df_train_file_cat = pd.merge(train_img_df, train_anns_df, on='image_id')

df_train_file_cat['category_id']=df_train_file_cat['category_id'].astype(str)

df_train_file_cat.head()
print(len(df_train_file_cat))
# splitting data into train and validation

from sklearn.model_selection import train_test_split

train, valid = train_test_split(df_train_file_cat, stratify=df_train_file_cat.category_id, test_size=0.2)
nb_classes = 1010

batch_size = 64

img_size = 299

nb_epochs = 2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add our data-augmentation parameters to ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255.,

                                   rotation_range = 20,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



# Note that the validation data should not be augmented!

test_datagen = ImageDataGenerator( rescale = 1.0/255. )



# Flow training images in batches of 20 using train_datagen generator

train_generator = train_datagen.flow_from_dataframe(train,

                                                    directory="../input/inaturalist-2019-fgvc6/train_val2019", 

                                                    x_col='file_name', 

                                                    y_col='category_id',

                                                    batch_size = batch_size,

                                                    class_mode = 'categorical', 

                                                    target_size = (img_size, img_size), 

                                                    color_mode='rgb')     



# Flow validation images in batches of 20 using test_datagen generator

validation_generator =  test_datagen.flow_from_dataframe( valid,

                                                          directory="../input/inaturalist-2019-fgvc6/train_val2019", 

                                                          x_col='file_name', 

                                                          y_col='category_id',

                                                          batch_size  = batch_size,

                                                          class_mode  = 'categorical', 

                                                          target_size = (img_size, img_size), 

                                                          color_mode='rgb')
from tensorflow.keras import layers

from tensorflow.keras import Model

  

from tensorflow.keras.applications.inception_v3 import InceptionV3



local_weights_file = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



pre_trained_model = InceptionV3(input_shape = (img_size, img_size, 3), 

                                include_top = False, 

                                weights = None)



pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:

    layer.trainable = True

for layer in pre_trained_model.layers:

  if layer.name == 'mixed6':

    break

  layer.trainable = False

  

# pre_trained_model.summary()



last_layer = pre_trained_model.get_layer('mixed7')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output
# Check the trainable status of the individual layers

for layer in pre_trained_model.layers:

    print(layer, layer.trainable)
from tensorflow.keras.optimizers import RMSprop



# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(rate=0.2)(x)                  

# Add a final sigmoid layer for classification

x = layers.Dense  (nb_classes, activation='softmax')(x)           



model = Model( pre_trained_model.input, x) 



model.compile(optimizer = RMSprop(lr=0.0001), 

              loss = 'categorical_crossentropy', 

              metrics = ['accuracy'])

history = model.fit_generator(train_generator, epochs=nb_epochs, validation_data = validation_generator, verbose = 1)



model.save("rps.h5")
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()