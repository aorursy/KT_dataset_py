# Imports 

import tensorflow as tf

import os

import cv2

import numpy as np

import matplotlib.pyplot as plt
def load_data(directory):

    directories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    labels = []

    images = []

    print(directories)

    for d in directories:

        subdir = os.path.join(directory, d)

        print(subdir)

        filenames = [ os.path.join(subdir, file_name) for file_name in os.listdir(subdir) if file_name.endswith('.ppm')]

        for file_name in filenames:

            images.append(cv2.imread(file_name))

            labels.append(int(d))

    return images, labels
Root = '../input/Belgium_TSC/'

train_directory = os.path.join(Root,'Training')

images, labels = load_data(train_directory)

images = np.array(images)

labels = np.array(labels)
print(images.ndim)

print(images.size)

print(images[0])
print(labels.ndim)

print(labels.size)

print(labels[0])
plt.hist(labels, 62)

plt.show()
traffic_signs = [300, 2250, 3650, 4000]



for i in range(len(traffic_signs)):

    plt.subplot(1, 4, i+1)

    plt.axis('off')

    plt.imshow(images[traffic_signs[i]])

    plt.subplots_adjust(wspace=0.5)

    plt.show()

    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 

                                                  images[traffic_signs[i]].min(), 

                                                  images[traffic_signs[i]].max()))

    

labels = list(labels)

unique_labels = set(labels)

plt.figure(figsize = (15,15))

print(unique_labels)

i = 1 



for label in unique_labels:

    image = images[labels.index(label)]

    plt.subplot(8,8, i)

    plt.axis('off')

    # Add a title to each subplot 

    plt.title("Label {0} ({1})".format(label, labels.count(label)))

    # Add 1 to the counter

    i += 1

    # And you plot this first image 

    plt.imshow(image)

plt.show()
images28 = [cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA) for image in images]
images28 = [cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY) for image in images28]

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)),

  tf.keras.layers.MaxPooling2D((2, 2)),

  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

  tf.keras.layers.MaxPooling2D((2, 2)),

  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(62, activation='softmax')

])

model.summary()

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
images28 = np.array(images28)

images_extend = np.expand_dims(images28, -1)

print(images_extend.shape)
model.fit(images_extend, np.array(labels), epochs=100)
test_directory = os.path.join(Root,'Testing')

test_images, test_labels = load_data(test_directory)
print(test_images[0].shape)
test_images = np.array(test_images)

test_labels = np.array(test_labels)
print(test_images[0].shape)
test_images = [cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA) for image in test_images]
test_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_images]
test_images = np.expand_dims(np.array(test_images), -1)

test_labels = np.array(test_labels)
model.evaluate(test_images, test_labels, verbose=2)