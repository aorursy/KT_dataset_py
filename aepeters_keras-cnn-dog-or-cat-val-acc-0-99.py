import numpy as np

import pandas as pd 

from keras.preprocessing.image import ImageDataGenerator,load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os

print(os.listdir("../input"))

import zipfile





zip_files = ['test1', 'train']

# Will unzip the files so that you can see them..

for zip_file in zip_files:

    with zipfile.ZipFile("../input/{}.zip".format(zip_file),"r") as z:

        z.extractall(".")

        print("{} unzipped".format(zip_file))
filenames = os.listdir("../working/train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
df.head()
df.tail()
df['category'].value_counts().plot.bar()
sample = random.choice(filenames)

image = load_img("../working/train/"+sample)

plt.imshow(image)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, ZeroPadding2D, Dropout, BatchNormalization

from tensorflow.keras import layers

from tensorflow.keras import Model

from tensorflow.keras.applications import inception_resnet_v2



IMG_SHAPE = (256,256, 3)





base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE,

                                               include_top=False,

                                               weights='imagenet')
from tensorflow.keras.optimizers import Adam

from tensorflow.python.keras import regularizers

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint 



base_model.trainable = True

print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards

fine_tune_at = 700



# Freeze all the layers before the `fine_tune_at` layer

for layer in base_model.layers[:fine_tune_at]:

  layer.trainable =  False



last_layer = base_model.get_layer('conv_7b_ac')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output



# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 500 hidden units and ReLU activation

x = layers.Dense(500, activation='relu',kernel_regularizer=regularizers.l2(0.1))(x)

# Add a dropout rate of 0.3

x = layers.Dropout(0.3)(x)   

x = layers.Dense(10, activation='relu',kernel_regularizer=regularizers.l2(0.1))(x)          

# Add a final sigmoid layer for classification

x = layers.Dense  (1, activation='sigmoid')(x)



model = Model( base_model.input, x) 

model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_acc', 

    verbose=1,

    patience=5,

    restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                             patience=3, min_lr=0.00000001,verbose = 1)
callbacks = [early_stopping, reduce_lr]
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 
train_df, validate_df = train_test_split(df, test_size=0.10, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size=256
train_datagen = ImageDataGenerator(

    rotation_range=35,

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.2,

    height_shift_range=0.2

)



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    "../working/train", 

    x_col='filename',

    y_col='category',

    target_size=(256,256),

    class_mode='binary',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "../working/train/", 

    x_col='filename',

    y_col='category',

    target_size=(256,256),

    class_mode='binary',

    batch_size=batch_size

)
from tensorflow.keras.optimizers import Adam

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=True)

model.compile(optimizer = opt, 

              loss = 'binary_crossentropy', 

              metrics = ['accuracy'])
example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example_df, 

    "../working/train", 

    x_col='filename',

    y_col='category',

    target_size=(256,256),

    class_mode='categorical'

)
plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
epochs=3

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=22500//256,

    steps_per_epoch=2500//256

)
model.save_weights("model.h5")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['acc'], color='b', label="Training accuracy")

ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()