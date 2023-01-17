# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

print ("done")
img_dir = '../input/image-classification/images/images'

val_dir = '../input/image-classification/validation/validation'
! pip install -q tf-nightly
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import datasets, layers, models, regularizers

tf.__version__
img_height = 256

img_width = 256

batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="training",

    seed=1007,

    shuffle=True,

    image_size=(img_height, img_width),

    batch_size=batch_size,

)
train_ds.class_names
val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="validation",

    seed=1007,

    shuffle=True,

    image_size=(img_height, img_width),

    batch_size=batch_size,

)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=val_dir,

    validation_split=0.9999,

    subset="validation",

    seed=1007,

    image_size=(img_height, img_width),

    batch_size=batch_size,

)
# put your code here 

class_names = train_ds.class_names



plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(16):

        ax = plt.subplot(4, 4, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")
epochs = 15



def create_model():

       

    class_numbers = len (class_names) 

    model = models.Sequential()

    

    model.add(layers.experimental.preprocessing.Rescaling(1./255))

    

    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height,img_height, 3)))

    model.add(layers.MaxPooling2D((2, 2)))

    

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

    

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

    

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

    

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

    

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(class_numbers, activation='softmax'))

    

    model.compile(optimizer='Adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])

    

    return model
model = create_model()
checkpoint_path = "training_1/cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)



# Create a callback that saves the model's weights

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,

                                                 save_weights_only=True,

                                                 verbose=1)

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)







# Train the model with the new callback

history = model.fit(train_ds, epochs=epochs, validation_data=(val_ds),

                    callbacks=[cp_callback,earlystop])  # Pass callback to training
# Loads the weights

model.load_weights(checkpoint_path)



# Re-evaluate the model

loss,acc = model.evaluate(test_ds, verbose=2)

print("Restored model, accuracy: {:5.2f}%".format(100*acc))
def plot_history(history, epochs):

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']



    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs_range = range(epochs+1)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)

    plt.plot(epochs_range, acc, label='Training Accuracy')

    plt.plot(epochs_range, val_acc, label='Validation Accuracy')

    plt.legend(loc='lower right')

    plt.xlabel('Number of epochs')

    plt.ylabel('Accuracy (%)')

    plt.title('Training and Validation Accuracy')



    plt.subplot(1, 2, 2)

    plt.plot(epochs_range, loss, label='Training Loss')

    plt.plot(epochs_range, val_loss, label='Validation Loss')

    plt.legend(loc='upper right')

    plt.xlabel('Number of epochs')

    plt.ylabel('Loss (%)')

    plt.title('Training and Validation Loss')

    plt.show()
plot_history(history, earlystop.stopped_epoch)
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
test_dir = '../input/image-classification/test/test'

os.listdir(test_dir)
test2_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=test_dir,

    validation_split=0.9999,

    subset="validation",

    seed=1007,

    image_size=(img_height, img_width),

    batch_size=batch_size,

)
predictions = model.predict(test2_ds)
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 10))

for images, labels in test2_ds:

    for i in range(0,9):

        ax = plt.subplot(3, 3, i + 1)

        predictions[i]

        score = tf.nn.softmax(predictions[i])

        percent_confidence = 100 * np.max(score)

        plt.title(class_names[np.argmax(score)]+' ('+str (round(percent_confidence,2))+' %)')

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.axis("off")
from skimage import io
# First Image

plt.figure(figsize=(16, 10))

image_url = "https://ochsner-craft.s3.amazonaws.com/blog/articles/_930x524_crop_center-center_75_none/Lemongrass-Tea-for-Digestive-Health.jpg"

image_path = tf.keras.utils.get_file('Lemongrass-Tea-for-Digestive-Health', origin=image_url)

img = keras.preprocessing.image.load_img(image_path, target_size=(256, 256))

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])

percent_confidence = 100 * np.max(score)

plt.title(class_names[np.argmax(score)]+' ('+str (round(percent_confidence,2))+' %)')

image = io.imread(image_url)

plt.imshow(image)

plt.axis("off")
# Second Image

plt.figure(figsize=(16, 10))

image_url = "https://www.diplomacy24.com/wp-content/uploads/2018/04/Experts-talk-on-importance-of-art-culture-in-diplomacy.jpg"

image_path = tf.keras.utils.get_file('Experts-talk-on-importance-of-art-culture-in-diplomacy', origin=image_url)

img = keras.preprocessing.image.load_img(image_path, target_size=(256, 256))

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])

percent_confidence = 100 * np.max(score)

plt.title(class_names[np.argmax(score)]+' ('+str (round(percent_confidence,2))+' %)')

image = io.imread(image_url)

plt.imshow(image)

plt.axis("off")
# Third Image

plt.figure(figsize=(16, 10))

image_url = "https://cdn.uniguide.com/wp-content/uploads/2020/09/Belize-741x486.jpg"

image_path = tf.keras.utils.get_file('Belize-741x486', origin=image_url)

img = keras.preprocessing.image.load_img(image_path, target_size=(256, 256))

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])

percent_confidence = 100 * np.max(score)

plt.title(class_names[np.argmax(score)]+' ('+str (round(percent_confidence,2))+' %)')

image = io.imread(image_url)

plt.imshow(image)

plt.axis("off")