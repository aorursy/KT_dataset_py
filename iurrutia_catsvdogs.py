# Reading in data to build a neural net cats vs dogs classifier on Kaggle data set using Keras


# For reading in data from kaggle (works on a)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from tensorflow import keras 

print(tf.__version__)
print(keras.__version__)
print(os.listdir("../input"))
# preparing data

train_dogs = "../input/dogs-cats-images/dataset/training_set/dogs"
train_cats = "../input/dogs-cats-images/dataset/training_set/cats"
test_dogs = "../input/dogs-cats-images/dataset/test_set/dogs"
test_cats = "../input/dogs-cats-images/dataset/test_set/cats"

len_train_d = len([name for name in os.listdir(train_dogs)])
len_train_c = len([name for name in os.listdir(train_cats)])
len_test_d = len([name for name in os.listdir(test_dogs)])
len_test_c = len([name for name in os.listdir(test_cats)])


print("Number of training images:", train_dogs+train_cats)
print("Number of test images:", test_dogs+test_cats)

# Check what one of these filenames looksprint(train_filenames[0].split('.'))

train_filenames = os.listdir(train_dogs)
print(train_filenames[0].split('.'))
# so the image name is indexed by 1, and the label is indexed by 0 (but we can also get the label via folder)
# creating df with train labels

# For dogs
train_filenames = os.listdir(train_dogs)
ids = []
label = []
for filename in train_filenames:
    label.append(filename.split('.')[0])
    ids.append(filename.split('.')[1])

train_df = pd.DataFrame({
    'id': ids,
    'label': label
})

# For cats:
train_filenames = os.listdir(train_cats)
ids = []
label = []
for filename in train_filenames:
    label.append(filename.split('.')[0])
    ids.append(filename.split('.')[1])

train_df2 = pd.DataFrame({
    'id': ids,
    'label': label
})

train_df = pd.concat([train_df, train_df2])
train_df.tail()
train_df.head()
# same but for test labels

# For dogs
test_filenames = os.listdir(test_dogs)
ids = []
label = []
for filename in test_filenames:
    label.append(filename.split('.')[0])
    ids.append(filename.split('.')[1])

test_df = pd.DataFrame({
    'id': ids,
    'label': label
})

# For cats:
test_filenames = os.listdir(test_cats)
ids = []
label = []
for filename in test_filenames:
    label.append(filename.split('.')[0])
    ids.append(filename.split('.')[1])

test_df2 = pd.DataFrame({
    'id': ids,
    'label': label
})

test_df = pd.concat([test_df, test_df2])

# Check:
test_df
# Read in data using https://www.kaggle.com/jsvishnuj/cats-dogs-classification-using-cnn
# https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
# ^ Interesting blog post on some of the image data generator works, with pictures


from tensorflow.keras.preprocessing.image import ImageDataGenerator


test_dir="../input/dogs-cats-images/dog vs cat/dataset/test_set"
train_dir="../input/dogs-cats-images/dog vs cat/dataset/training_set"

train_dir_cats = train_dir + '/cats'
train_dir_dogs = train_dir + '/dogs'
test_dir_cats = test_dir + '/cats'
test_dir_dogs = test_dir + '/dogs'

data_generator = ImageDataGenerator(rescale = 1.0/255.0, zoom_range = 0.2)
batch_size = 32

training_data = data_generator.flow_from_directory(directory = train_dir,
                                                   color_mode="grayscale",
                                                   target_size = (64, 64),
                                                   batch_size = batch_size,
                                                   class_mode = 'binary')
testing_data = data_generator.flow_from_directory(directory = test_dir,
                                                  color_mode="grayscale",
                                                  target_size = (64, 64),
                                                  batch_size = batch_size,
                                                  class_mode = 'binary')

# Code from the textbook:

for image,label in testing_data:
    print("shape:", image.shape)
    print(image, label)
    break
model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300, activation = "relu")) # figure out size of layers
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(2, activation = "sigmoid"))
model.build(input_shape = (64,64))
model.summary()
print(model.layers)
hidden1 = model.layers[1]
print(hidden1.name)
model.get_layer('dense_3') is hidden1
weights, biases = hidden1.get_weights()
print(weights.shape)
print("WEIGHTS before initializing:", weights)
print(biases.shape)
print(biases)
# Also try loss=tf.keras.losses.BinaryCrossentropy() or loss = "categorical_crossentropy"
model.compile(loss = "binary_crossentropy",optimizer = "sgd", metrics = ["accuracy"])
print(weights.shape)
print("WEIGHTS after initializing:", weights)
history = model.fit(X_train, y_train, epochs = 30, validation_data = (x_valid,y_valid))