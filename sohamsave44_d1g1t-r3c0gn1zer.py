# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from sklearn.model_selection import train_test_split
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
test_data.head()
X_all_train = train_data.drop(['label'], axis = 1)
X_all_train.head()
Y_all_train = train_data["label"]
Y_all_train.head()
X_all_train = X_all_train / 255
X_all_train = X_all_train.values.reshape(-1,28,28,1)
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_all_train, Y_all_train, test_size = 0.1, random_state=random_seed)
X_test = test_data
X_test.head()
X_test = X_test / 255
X_test = X_test.values.reshape(-1,28,28,1)
input_size = 784
output_size = 10

hidden_layer_size = 5000

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # input layer
    
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 3rd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 4th hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 5th hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 6th hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 7th hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 8th hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 9th hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 10th hidden layer
    
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
NUM_EPOCHS = 30
batch_size = 150


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
model.fit(datagen.flow(X_train,Y_train,batch_size = batch_size), epochs=NUM_EPOCHS, validation_data=(X_val, Y_val), verbose =2)
results = model.predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
output = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

output.to_csv("submission.csv",index=False)
