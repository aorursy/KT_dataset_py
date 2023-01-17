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
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import os

import numpy as np

import matplotlib.pyplot as plt
tf.__version__
import zipfile

with zipfile.ZipFile('../input/platesv2/plates.zip', 'r') as zip_obj:

   # Extract all the contents of zip file in current directory

   zip_obj.extractall('/kaggle/working/')

    

print('After zip extraction:')

print(os.listdir("/kaggle/working/"))
! ls plates/train/
! ls plates/train/cleaned | head 
! ls plates/train/dirty | head 
PATH =  'plates'

train_dir = os.path.join(PATH, 'train')

train_dir
# directory with our training dirty pictures

train_dirty_dir = os.path.join(train_dir, 'dirty')  

# directory with our training cleaned pictures

train_cleaned_dir = os.path.join(train_dir, 'cleaned')  

train_dirty_dir, train_cleaned_dir
num_dirty_tr = len(os.listdir(train_dirty_dir))

num_cleaned_tr = len(os.listdir(train_cleaned_dir))

num_dirty_tr, num_cleaned_tr
total_train = num_dirty_tr + num_cleaned_tr

total_train
batch_size = 10

epochs = 200

IMG_SIZE = 160
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.

def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
image_gen_train = ImageDataGenerator(

                    rescale=1./255,

                    rotation_range=15,

                    width_shift_range=.1,

                    height_shift_range=.1,

                    horizontal_flip=True,

                    zoom_range=0.1, 

                    brightness_range=[0.8,1.0]

                    )
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,

                                                     directory=train_dir,

                                                     shuffle=True,

                                                     target_size=(IMG_SIZE, IMG_SIZE),

                                                     class_mode='binary', 

)
sample_training_images, _ = next(train_data_gen)

plotImages(sample_training_images[:5])
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
IMG_SHAPE=(IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2

base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,

                                               include_top=False,

                                               weights='imagenet')
base_model.trainable = False
# Let's take a look at the base model architecture

base_model.summary()
image_batch = sample_training_images[:batch_size]

image_batch.shape
feature_batch = base_model(image_batch)

print(feature_batch.shape)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

feature_batch_average = global_average_layer(feature_batch)

print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(1)

prediction_batch = prediction_layer(feature_batch_average)

print(prediction_batch.shape)
model = tf.keras.Sequential([

  base_model,

  global_average_layer,

  prediction_layer

])



base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])



model.summary()
len(model.trainable_variables)


# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

# mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)



# fit model

history = model.fit(train_data_gen,

                    epochs=400, 

                    callbacks=[es, mc]

                    )
saved_model = tf.keras.models.load_model('best_model.h5')
! ls plates/test/ | head 
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(  

        'plates',

        classes=['test'],

        target_size = (IMG_SIZE, IMG_SIZE),

        batch_size = 1,

        shuffle = False,        

        class_mode = None)  
test_generator.reset()

predict = saved_model.predict_generator(test_generator, steps = len(test_generator.filenames))

len(predict)
sub_df = pd.read_csv('../input/platesv2/sample_submission.csv')

sub_df.head()
sub_df['label'] = predict

sub_df['label'] = sub_df['label'].apply(lambda x: 'dirty' if x > 0.5 else 'cleaned')

sub_df.head()
sub_df['label'].value_counts()
sub_df.to_csv('sub.csv', index=False)