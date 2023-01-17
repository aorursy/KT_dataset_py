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
batch_size = 200
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
# put your code here 
plt.figure(figsize=(20, 20))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# put your code here 
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
epochs=7
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
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
# put your code here 
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
test_dir = '../input/image-classification/test/test/classify'
os.listdir(test_dir)
from keras.preprocessing import image
sunflower_path = "../input/image-classification/test/test/classify/1.JPG"

img = keras.preprocessing.image.load_img(sunflower_path, target_size=(img_height, img_width))
                                         
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(train_ds.class_names[np.argmax(score)], 100 * np.max(score))
)

plt.figure(figsize=(5, 5))
plt.imshow(img_array[0].numpy().astype("uint8"))
plt.axis("off")
# put your code here 
sunflower_url = "https://static.scientificamerican.com/?w=590&h=800&ACB6A419-81EB-4B9C-B846FD8EBFB16FBE"
sunflower_path = tf.keras.utils.get_file('ima', origin=sunflower_url)
img = keras.preprocessing.image.load_img(sunflower_path, target_size=(img_height, img_width))
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
plt.axis("off")