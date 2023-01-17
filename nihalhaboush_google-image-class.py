# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img_dir = '../input/image-classification/images/images'

val_dir = '../input/image-classification/validation/validation'
! pip install -q tf-nightly
import tensorflow as tf

from tensorflow import keras

tf.__version__
# change as you want

image_size = (180, 180)

img_height = 180

img_width = 180

batch_size = 64
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="training",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

)
train_ds.class_names
val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="validation",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=val_dir,

    validation_split=0.9999,

    subset="validation",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

)
class_names = train_ds.class_names

print(class_names)
# put your code here 

import matplotlib.pyplot as plt



plt.figure(figsize=(16, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[int(labels[i])])

        plt.axis("off")
for image_batch, labels_batch in train_ds:

  print(image_batch.shape)

  print(labels_batch.shape)

  break
AUTOTUNE = tf.data.experimental.AUTOTUNE



train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
# put your code here 

import PIL

import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

num_classes = 4



model = Sequential([

  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

  layers.Conv2D(16, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Flatten(),

  layers.Dense(128, activation='relu'),

  layers.Dense(num_classes)

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.summary()
epochs=8

history = model.fit(

  train_ds,

  validation_data=val_ds,

  epochs=epochs

)
# put your code here 

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss=history.history['loss']

val_loss=history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
data_augmentation = keras.Sequential(

  [

    layers.experimental.preprocessing.RandomFlip("horizontal", 

                                                 input_shape=(img_height, 

                                                              img_width,

                                                              3)),

    layers.experimental.preprocessing.RandomRotation(0.1),

    layers.experimental.preprocessing.RandomZoom(0.1),

  ]

)
plt.figure(figsize=(10, 10))

for images, _ in train_ds.take(1):

  for i in range(9):

    augmented_images = data_augmentation(images)

    ax = plt.subplot(3, 3, i + 1)

    plt.imshow(augmented_images[0].numpy().astype("uint8"))

    plt.axis("off")
''''model = Sequential([

  data_augmentation,

  layers.experimental.preprocessing.Rescaling(1./255),

  layers.Conv2D(16, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Dropout(0.2),

  layers.Flatten(),

  layers.Dense(128, activation='relu'),

  layers.Dense(num_classes)

])'''
''''model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])'''
''''model.summary()'''
''''epochs = 8

history = model.fit(

  train_ds,

  validation_data=val_ds,

  epochs=epochs

)'''
''''acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()'''
# put your code here 

!pip install -q pyyaml h5py  # Required to save models in HDF5 format
import os



import tensorflow as tf

from tensorflow import keras



print(tf.version.VERSION)
# Save the entire model as a SavedModel.

!mkdir -p saved_model

model.save('saved_model/my_model') 
# my_model directory

!ls saved_model



# Contains an assets folder, saved_model.pb, and variables folder.

!ls saved_model/my_model
new_model = tf.keras.models.load_model('saved_model/my_model')



# Check its architecture

new_model.summary()
test_dir = '../input/image-classification/test/test/classify'

os.listdir(test_dir)
# put your code here 

probability_model = tf.keras.Sequential([model, 

                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_ds)
predictions[0]
np.argmax(predictions[0])
images[0].numpy().astype("uint8")

     
class_names[np.argmax(predictions[0])]
predictions[2]
class_names[np.argmax(predictions[2])]
predictions[6]
class_names[np.argmax(predictions[6])]
from keras.preprocessing import image



plt.figure(figsize=(16, 16))

for i in range(10):

    img = tf.keras.preprocessing.image.load_img(test_dir+'/'+os.listdir(test_dir)[i],target_size=image_size)

    ax = plt.subplot(3, 4, i + 1)

    plt.imshow(img)

    

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis = 0)

    predction= np.argmax(model.predict(img))

    plt.title(class_names[predction])

    #print('image class is :',i,class_names[predction])

    plt.title(class_names[predction])

    plt.axis("off")
url = 'https://upload.wikimedia.org/wikipedia/commons/d/d8/MoscowMuseum_of_Architecture.JPG'

path = tf.keras.utils.get_file('MoscowMuseum_of_Architecture', origin=url)



img = keras.preprocessing.image.load_img(

    path, target_size=(img_height, img_width)

)

plt.figure(figsize=(10, 10))

plt.imshow(img)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score))

)
url = 'https://images2.minutemediacdn.com/image/upload/c_crop,h_1126,w_2000,x_0,y_181/f_auto,q_auto,w_1100/v1554932288/shape/mentalfloss/12531-istock-637790866.jpg'

path = tf.keras.utils.get_file('12531-istock-637790866', origin=url)



img = keras.preprocessing.image.load_img(

    path, target_size=(img_height, img_width)

)

plt.figure(figsize=(10, 10))

plt.imshow(img)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score))

)
url = 'https://news.artnet.com/app/news-upload/2015/08/mona-lisa_2-e1440166499927.jpg'

path = tf.keras.utils.get_file('12531-istock-637790866', origin=url)



img = keras.preprocessing.image.load_img(

    path, target_size=(img_height, img_width)

)

plt.figure(figsize=(10, 10))

plt.imshow(img)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score))

)