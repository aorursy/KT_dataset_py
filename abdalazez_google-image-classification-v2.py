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

        break

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img_dir = '../input/image-classification/images/images'

val_dir = '../input/image-classification/validation/validation'
! pip install -q tf-nightly
import matplotlib.pyplot as plt

import numpy as np

import os

import PIL

import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

tf.__version__
# change as you want

image_size = (180, 180)

img_height = 180

img_width = 180

batch_size = 32

#32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="training",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

    shuffle=True

)
train_ds.class_names
val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="validation",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

    shuffle=True

)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=val_dir,

    validation_split=0.9999,

    subset="validation",

    seed=1007,

    image_size=image_size,

    batch_size=batch_size,

    shuffle=True

)
# put your code here 

import matplotlib.pyplot as plt



plt.figure(figsize=(15, 15))

for images, labels in train_ds.take(1):

  for i in range(9):

    ax = plt.subplot(3, 3, i + 1)

    plt.imshow(images[i].numpy().astype("uint8"))

    plt.title(train_ds.class_names[labels[i]])

    plt.axis("off")
for image_batch, labels_batch in train_ds:

  print(image_batch.shape)

  print(labels_batch.shape)

  break
# put your code here

num_classes = 4



model = Sequential([

  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

  layers.Conv2D(4, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Dropout(0.2),

  layers.Conv2D(8, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Dropout(0.2),

  layers.Conv2D(16, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Dropout(0.2),

  layers.Conv2D(32, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Dropout(0.2),

  layers.Conv2D(64, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Dropout(0.2),

  layers.Flatten(),

  layers.Dense(128, activation='relu'),

  layers.Dense(num_classes)

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.summary()
epochs=30

history = model.fit(

  train_ds,

  validation_data=val_ds,

  epochs=epochs,

  #batch_size = batch_size

)
print('Done')
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
!mkdir -p saved_modelimg

model.save('saved_modelimg/my_modelimg') 
# my_model directory

!ls saved_modelimg



# Contains an assets folder, saved_model.pb, and variables folder.

!ls saved_modelimg/my_modelimg
new_model = tf.keras.models.load_model('saved_modelimg/my_modelimg')



# Check its architecture

new_model.summary()
test_dir = '../input/image-classification/test/test/classify'

os.listdir(test_dir)
# put your code here 

from keras.preprocessing import image

list_img = os.listdir(test_dir)

imgIs =  '../input/image-classification/test/test/classify/'+list_img[1]

test_image = []#image.load_img(imgIs, target_size=(img_height, img_width))

info = []

for i in list_img:

    test_image = image.load_img('../input/image-classification/test/test/classify/'+i, target_size=(img_height, img_width))

    img_array = keras.preprocessing.image.img_to_array(test_image)

    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    info.append([img_array,score])

    

j=0

for images, labels in info:

    plt.figure(figsize=(15, 15))

    ax = plt.subplot(3, 3, j + 1)

    plt.imshow(images[0].numpy().astype("uint8"))

    plt.title("This image most likely belongs to {} with a {:.2f} percent confidence.".format(train_ds.class_names[np.argmax(labels)], 100 * np.max(score)))

    plt.axis("off")
# put your code here 



##sunflower_url2 = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"

sunflower_url = "https://upload.wikimedia.org/wikipedia/commons/6/6a/Mona_Lisa.jpg"



sunflower_path = tf.keras.utils.get_file('image11', origin=sunflower_url)



img = keras.preprocessing.image.load_img(

    sunflower_path, target_size=(img_height, img_width)

)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(train_ds.class_names[np.argmax(score)], 100 * np.max(score))

)



plt.figure(figsize=(5, 5))

plt.imshow(img_array[0].numpy().astype("uint8"))

#plt.title(train_ds.class_names[labels[i]])

plt.axis("off")