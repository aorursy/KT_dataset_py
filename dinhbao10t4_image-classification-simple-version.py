import numpy as np # linear algebra
import pandas as pd # I/O of data
import matplotlib.pyplot as plt # making plots
%matplotlib inline
# keras for deep learning
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
#from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
#from sklearn.model_selection import train_test_split
from skimage import io
import os
import scipy.misc
from scipy.misc import imread, imresize
import csv
import random, shutil, zlib # directory operations
from glob import glob
print('Current directory is {}'.format(os.getcwd())) # print the path of the current directory
print('Current directory contains the follwoing sub-directories:\n {}'.format(os.listdir())) # print the contents of the current directory
# First, look at everything.
from subprocess import check_output
# print(check_output(["ls", "../input/fruits-360_dataset/fruits-360/Training"]).decode("utf8"))
# Training and testing folders
train_path = '../input/fruits-360_dataset/fruits-360/Training'
test_path = '../input/fruits-360_dataset/fruits-360/Test'
# Get train and test files
image_files = glob(train_path + '/*/*.jp*g')
test_image_files = glob(test_path + '/*/*.jp*g')
print(np.random.choice(test_image_files))
# Get number of classes
folders = glob(train_path + '/*')

# Display any random image
plt.imshow(plt.imread(np.random.choice(image_files)))
plt.axis('off')
plt.show()
# Resize all the images to this
IMAGE_SIZE = [100, 100]
# Training config
batch_size = 32
# Create an instance of ImageDataGenerator
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  rescale=1./255
)

# Get label mapping of class and label number
test_gen = gen.flow_from_directory(test_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
    labels[v] = k
# Create generators for training and validation
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  test_path,
  target_size=IMAGE_SIZE,
  shuffle=False,
  batch_size=batch_size,
)

output_units = len(test_gen.class_indices)
print(output_units)
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (100, 100, 3), padding='same'))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu', padding='same'))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# taking the 2-D array, i.e pooled image pixels and converting them to a one dimensional single vector.
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu', kernel_initializer='he_normal'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = output_units, activation = 'softmax', kernel_initializer='he_normal'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
epochs_val = 3
validation_steps_val = len(test_image_files) // batch_size
steps_per_epoch_val = len(image_files) // batch_size

#steps_per_epoch: Integer. Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and 
#starting the next epoch. It should typically be equal to the number of samples of your dataset divided by the batch size. 
classifier.fit_generator(train_generator, steps_per_epoch = steps_per_epoch_val, epochs = epochs_val, validation_data = valid_generator, validation_steps = validation_steps_val)
# Visualizing predictions
result = np.round(classifier.predict_generator(valid_generator))
import random
test_files = []
actual_res = []
test_res = []
for i in range(0, 5):
    rng = random.randint(0, len(valid_generator.filenames))
    test_files.append(test_path + '/' +  valid_generator.filenames[rng])
    actual_res.append(valid_generator.filenames[rng].split('/')[0])
    test_res.append(labels[np.argmax(result[rng])])
for i in range(0, 5):
    plt.imshow(plt.imread(test_files[i]))
    plt.axis('off')
    plt.show()
    print("Actual class: " + str(actual_res[i]))
    print("Predicted class: " + str(test_res[i]))
