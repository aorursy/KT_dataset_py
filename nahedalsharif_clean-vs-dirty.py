# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!unzip -q /kaggle/input/platesv2/plates.zip
!ls
! pip install tf_nightly
import tensorflow as tf

tf.__version__
batch_size = 32

img_height = 200

img_width = 200



image_size=(200, 200)

print('done')
train_ds = tf.keras.preprocessing.image_dataset_from_directory( "plates/train",

    validation_split=0.2,

    subset="training",

    seed=1337,

    image_size=image_size,

    batch_size=batch_size,

)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    "plates/train",

    validation_split=0.2,

    subset="validation",

    seed=1337,

    image_size=image_size,

    batch_size=batch_size,

)
import matplotlib.pyplot as plt



plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(int(labels[i]))

        plt.axis("off")
for image_batch, labels_batch in train_ds:

    print(image_batch.shape)

    print(labels_batch.shape)

    break
from tensorflow import keras

from tensorflow.keras import datasets, layers, models

from tensorflow.keras.models import Sequential



data_augmentation = keras.Sequential(

  [

    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),

    layers.experimental.preprocessing.RandomRotation(0.1),

    layers.experimental.preprocessing.RandomZoom(0.1),

    layers.experimental.preprocessing.RandomContrast(0.3),

   

  ]

)
plt.figure(figsize=(10, 10))

for images, _ in train_ds.take(1):

    for i in range(9):

        augmented_images = data_augmentation(images)

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(augmented_images[0].numpy().astype("uint8"))

        plt.axis("off")
#train_ds = train_ds.prefetch(buffer_size=32)

#val_ds = val_ds.prefetch(buffer_size=32)
num_classes=2

model = Sequential([

  data_augmentation,

  

layers.experimental.preprocessing.Rescaling(1./255),

  

layers.Conv2D(16, 3, padding='same', activation='relu'),

layers.MaxPooling2D(),

layers.Dropout(0.2),

    

layers.Conv2D(16, 3, padding='same', activation='relu'),

layers.MaxPooling2D(),



 

 

layers.Flatten(),

layers.Dense(32, activation='relu'),

layers.Dense(num_classes)

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.summary()

model_cp = keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)



epochs = 15

history = model.fit(

  train_ds,

  validation_data=val_ds,

  epochs=epochs,

    callbacks=[ model_cp, earlystop]

)
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(  

        'plates',

        classes=['test'],

        target_size = (200, 200),

        batch_size = 1,

        shuffle = False,        

        class_mode = None)  
test_generator.reset()

predict = model.predict_generator(test_generator, steps = len(test_generator.filenames))

len(predict)
import pandas as pd

sub_df = pd.read_csv('../input/platesv2/sample_submission.csv')

sub_df.head()
sub_df.label.value_counts()
sub_df['label'] = predict

sub_df['label'] = sub_df['label'].apply(lambda x: 'dirty' if x > 0.5 else 'cleaned')

sub_df.head()
sub_df.label.value_counts()
sub_df.to_csv('sub.csv', index=False)