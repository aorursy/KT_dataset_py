import numpy as np

import pandas as pd

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os

print(os.listdir("../input"))
#Set the input directory

input_dir_name="../input/cat-and-dog/training_set/training_set/"
import glob

dog_list=[]

cat_list=[]

for name in glob.glob(input_dir_name+"dogs/*"):

    dog_list.append(name)

for name in glob.glob(input_dir_name+"cats/*"):

    cat_list.append(name)
samples=random.choices(dog_list, k=6)

w=10

h=10

fig=plt.figure(figsize=(8, 8))

columns = 3

rows = 2

for i in range(1, columns*rows):

    fig.add_subplot(rows, columns, i)

    image = load_img(samples[i])

    plt.imshow(image)

plt.show()
batch_size = 128

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator

train_generator = train_datagen.flow_from_directory(

        input_dir_name,  # This is the source directory for training images

        target_size=(200, 200),  # All images will be resized to 200 x 200

        batch_size=batch_size,

        # Specify the classes explicitly

        classes = ['dogs','cats'],

        # Since we use categorical_crossentropy loss, we need categorical labels

        class_mode='categorical')
model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color

    # The first convolution

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The fourth convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The fifth convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a dense layer

    tf.keras.layers.Flatten(),

    # 128 neuron in the fully-connected layer

    tf.keras.layers.Dense(128, activation='relu'),

    # 2 output neurons for 2 classes with the softmax activation

    tf.keras.layers.Dense(2, activation='softmax')

])
model.summary()
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(lr=0.001),

              metrics=['acc'])
demo_run=False

n_epochs=3 if demo_run else 50

total_sample=train_generator.n

history = model.fit_generator(

        train_generator, 

        steps_per_epoch=int(total_sample/batch_size),  

        epochs=n_epochs,

        verbose=1)
model.save_weights("model.h5")
plt.plot(history.history['loss'], color='b', label="Training loss")

plt.show()
test_dir_name="../input/cat-and-dog/test_set/test_set"

os.listdir(test_dir_name)
test_dir_name="../input/cat-and-dog/test_set/test_set/"
test_datagen=ImageDataGenerator(rescale=1/255)

test_generator = test_datagen.flow_from_directory(

        test_dir_name,  # This is the source directory for test images

        target_size=(200, 200),  # All images will be resized to 200 x 200

        batch_size=batch_size,

        class_mode=None

    )
nb_samples=test_generator.n
test_generator.labels
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
predicted_labels= np.argmax(predict, axis=-1)
true_labels=test_generator.labels
from sklearn.metrics import accuracy_score

print(accuracy_score(predicted_labels,true_labels))
filename=test_generator.filenames[1]

label=predicted_labels[1]

plt.figure(figsize=(5, 5))

img = load_img(test_dir_name+filename)

plt.imshow(img)

if(label==0):

    cl="cat"

else:

    cl='dog'

plt.xlabel('Your predicted images is '+'(' + "{}".format(cl) + ')' )

plt.show()
test_generator.class_indices.items()