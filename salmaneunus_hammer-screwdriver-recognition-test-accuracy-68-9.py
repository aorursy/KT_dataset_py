!pip install tensorflow

import os
# Directory with our training horse pictures

train_hammer_dir = os.path.join('../input/mechanical-tools-dataset/train_data/train_data/Hammer')



# Directory with our training human pictures

train_Screwdriver_dir = os.path.join('../input/mechanical-tools-dataset/train_data/train_data/Screw Driver')
train_hammer_names = os.listdir(train_hammer_dir)

print(train_hammer_names[:20])



train_screw_names = os.listdir(train_Screwdriver_dir)

print(train_screw_names[:20])
print('total training hammer images:', len(os.listdir(train_hammer_dir)))

print('total training screwdriver images:', len(os.listdir(train_Screwdriver_dir)))
%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



# Parameters for our graph; we'll output images in a 10x10 configuration

nrows = 10

ncols = 10



# Index for iterating over images

pic_index = 0
# Set up matplotlib fig, and size it to fit 4x4 pics

fig = plt.gcf()

fig.set_size_inches(ncols * 4, nrows * 4)



pic_index += 200

next_hammer_pix = [os.path.join(train_hammer_dir, fname) 

                for fname in train_hammer_names[pic_index-50:pic_index]]

next_driver_pix = [os.path.join(train_Screwdriver_dir, fname) 

                for fname in train_screw_names[pic_index-50:pic_index]]



for i, img_path in enumerate(next_hammer_pix+next_driver_pix):

  # Set up subplot; subplot indices start at 1

  sp = plt.subplot(nrows, ncols, i + 1)

  sp.axis('Off') # Don't show axes (or gridlines)



  img = mpimg.imread(img_path)

  plt.imshow(img)



plt.show()
import tensorflow as tf
model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 300x300 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.25),

    # The second convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    # The third convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    # The fourth convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    # The fifth convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.25),

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for class ('hammer') and 1 for the other ('screwdriver')

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()
from tensorflow.keras.optimizers import RMSprop



model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=0.001),

              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1/255)



# Flow training images in batches of 128 using train_datagen generator

train_generator = train_datagen.flow_from_directory(

        '../input/mechanical-tools-dataset/train_data/train_data',  # This is the source directory for training images

        target_size=(300, 300),  # All images will be resized to 150x150

        batch_size=128,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')
history = model.fit(

      train_generator,

      steps_per_epoch=10,  

      epochs=40,

      verbose=2)
import cv2

import numpy as np

img = cv2.imread('../input/mechanical-tools-dataset/test_data/test_data/Hammer (1891).JPEG')

plt.imshow(img)

img = cv2.resize(img,(300,300))

img = np.reshape(img,[1,300,300,3])



classes = model.predict_classes(img)



print(classes)

if classes>0.5:

    print("This is a Screw Driver")

else:

    print("This is a Hammer")
img = cv2.imread('../input/mechanical-tools-dataset/test_data/test_data/Screwdriver (1421).JPEG')

plt.imshow(img)

img = cv2.resize(img,(300,300))

img = np.reshape(img,[1,300,300,3])



classes = model.predict_classes(img)



print(classes)

if classes>0.5:

    print("This is a Screw Driver")

else:

    print("This is a Hammer")
test_datagen = ImageDataGenerator(rescale=1/255)

test_generator = test_datagen.flow_from_directory(

        '../input/mechanical-tools-dataset/test_data/',  # This is the source directory for training images

        target_size=(300, 300),  # All images will be resized to 150x150

        batch_size=128,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')
model.predict_classes(test_generator)
scores = model.evaluate(test_generator, verbose=2)

print("Accuracy: %.2f%%" % (scores[1]*100))
import numpy as np

import random

from tensorflow.keras.preprocessing.image import img_to_array, load_img



# Let's define a new Model that will take an image as input, and will output

# intermediate representations for all layers in the previous model after

# the first.

successive_outputs = [layer.output for layer in model.layers[1:]]

#visualization_model = Model(img_input, successive_outputs)

visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# Let's prepare a random input image from the training set.

hammer_img_files = [os.path.join(train_hammer_dir, f) for f in train_hammer_names]

screw_img_files = [os.path.join(train_Screwdriver_dir, f) for f in train_screw_names]

img_path = random.choice(hammer_img_files + screw_img_files)



img = load_img(img_path, target_size=(300, 300))  # this is a PIL image

x = img_to_array(img)  # Numpy array with shape (150, 150, 3)

x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)



# Rescale by 1/255

x /= 255



# Let's run our image through our network, thus obtaining all

# intermediate representations for this image.

successive_feature_maps = visualization_model.predict(x)



# These are the names of the layers, so can have them as part of our plot

layer_names = [layer.name for layer in model.layers[1:]]



# Now let's display our representations

for layer_name, feature_map in zip(layer_names, successive_feature_maps):

    if len(feature_map.shape) == 4:

    # Just do this for the conv / maxpool layers, not the fully-connected layers

        n_features = feature_map.shape[-1]  # number of features in feature map

    # The feature map has shape (1, size, size, n_features)

        size = feature_map.shape[1]

    # We will tile our images in this matrix

        display_grid = np.zeros((size, size * n_features))

        for i in range(n_features):

      # Postprocess the feature to make it visually palatable

            x = feature_map[0, :, :, i]

            x -= x.mean()

            x /= x.std()

            x *= 64

            x += 128

            x = np.clip(x, 0, 255).astype('uint8')

      # We'll tile each filter into this big horizontal grid

            display_grid[:, i * size : (i + 1) * size] = x

    # Display the grid

        scale = 20. / n_features

        plt.figure(figsize=(scale * n_features, scale))

        plt.title(layer_name)

        plt.grid(False)

        plt.imshow(display_grid, aspect='auto', cmap='viridis')