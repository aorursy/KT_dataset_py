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
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

def get_data(filename):
    with open(filename) as training_file:
      # Your code starts here
        csv_reader = csv.reader(training_file, delimiter=',')
        line = 0
        images = []
        labels = []
        for row in csv_reader:
            if(line > 0):
                labels.append(row[0])
                images.append(row[1:785])
            line += 1
        images = np.array(images)
        images = np.reshape(images.astype(np.int32),(-1,28,28))
        labels = np.array(labels).astype(np.int32)
      # Your code ends here
    return images, labels

path_sign_mnist_train = "/kaggle/input/sign-language-mnist/sign_mnist_train.csv"
path_sign_mnist_test = "/kaggle/input/sign-language-mnist/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)
training_images_exp = np.expand_dims(training_images, axis=3)
testing_images_exp = np.expand_dims(testing_images, axis=3)

train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest")

validation_datagen = ImageDataGenerator(
    rescale=1./255.
    )
    
# Keep These
print(training_images_exp.shape)
print(testing_images_exp.shape)

print(training_labels.shape)
print(training_labels)

#training_labels_cat = tf.keras.utils.to_categorical(training_labels)
#print(training_labels_cat)
#print(training_labels_cat.shape)

#testing_labels_cat = tf.keras.utils.to_categorical(testing_labels)
#print(testing_labels_cat)
#print(testing_labels_cat.shape)

# using LabelBinarizer
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
training_labels_cat = label_binarizer.fit_transform(training_labels)
testing_labels_cat = label_binarizer.fit_transform(testing_labels)
print(training_labels_cat)
print(training_labels_cat.shape)
print(testing_labels_cat)
print(testing_labels_cat.shape)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(75, (3,3), strides=1, padding="same", activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"),
    tf.keras.layers.Conv2D(50, (3,3), strides=1, padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"),
    tf.keras.layers.Conv2D(25, (3,3), strides=1, padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"),                           
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(24, activation='softmax')
    ]
)

# Compile Model. 
model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])# Your Code Here)

# Train the Model
model.summary()

training_generator = train_datagen.flow(training_images_exp, training_labels_cat, batch_size=19)
validation_generator = validation_datagen.flow(testing_images_exp, testing_labels_cat, batch_size=22)

%matplotlib inline
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
for i in range(6):
    plt.subplot(2,3,i+1)
    for x,y in training_generator:
        plt.imshow((x[0]/255).reshape(28,28),cmap='gray')
        plt.title('y={}'.format(i+1))
        plt.axis('off')
        break
plt.tight_layout()
plt.show()

history = model.fit_generator(training_generator, 
                              steps_per_epoch=len(training_images_exp) // 19,
                              validation_data=validation_generator,
                              validation_steps=len(testing_images_exp) // 22,
                              epochs=5, verbose=1)# Your Code Here (set 'epochs' = 2))

model.evaluate(testing_images_exp / 255, testing_labels_cat, verbose=1)