# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img_dir = '../input/image-classification/images/images'

val_dir = '../input/image-classification/validation/validation'
! pip install -q tf-nightly
import tensorflow as tf

from tensorflow import keras

tf.__version__
from tensorflow import keras

from tensorflow.keras import datasets, layers, models

from tensorflow.keras.models import Sequential
# change as you want

image_size = (180, 180)

batch_size = 64

img_height = 180

img_width = 180
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
import matplotlib.pyplot as plt



plt.figure(figsize=(16, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(train_ds.class_names[int(labels[i])])

        plt.axis("off")
#AUTOTUNE = tf.data.experimental.AUTOTUNE



#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

image_batch, labels_batch = next(iter(normalized_ds))

first_image = image_batch[0]

# Notice the pixels values are now in `[0,1]`.

print(np.min(first_image), np.max(first_image)) 
from tensorflow.keras import datasets, layers, models



num_classes = 5



model = Sequential([

  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

  layers.Conv2D(32, 3, padding='same', activation='relu'),

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
epochs=10

history = model.fit(

  train_ds,

  validation_data=val_ds,

  epochs=epochs

)
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
model.save("my_model.h5")

print("model saved!!!")
test_dir = '../input/image-classification/test/test/classify'

os.listdir(test_dir)
pic_url = "../input/image-classification/test/test/classify/4.JPG"





img = keras.preprocessing.image.load_img(

    pic_url, target_size=(img_height, img_width)

)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0)



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(test_ds.class_names[np.argmax(score)], 100 * np.max(score))

)
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
url = 'https://beforetravelling.com/wp-content/uploads/2020/01/Adventure-Travel-1.jpg'

path = tf.keras.utils.get_file('Adventure-Travel-1', origin=url)

plt.figure(figsize=(7, 7))

img = keras.preprocessing.image.load_img(

    path, target_size=(img_height, img_width)

)

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

plt.figure(figsize=(7, 7))

plt.imshow(img)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score))

)