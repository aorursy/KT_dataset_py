import tensorflow as tf

from tensorflow import keras

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline
keras.__version__
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Load the training data and the testing data from the fashionmnist dataset

# which is mounted under ../input/fashionmnist

#

data_train_file = "../input/fashionmnist/fashion-mnist_train.csv"

data_test_file = "../input/fashionmnist/fashion-mnist_test.csv"



df_train = pd.read_csv(data_train_file)

df_test = pd.read_csv(data_test_file)



df_train.head()
def get_features_labels(df):

    # select all columns by the first column as that is the label column

    # normalize the data to be between 0-1 by dividing by 255

    # since the greyscale ranges from 0 to 255

    features = df.values[:,1:]/255

    

    #load the labels for this dataframe

    labels = df['label'].values

    return features, labels
train_features, train_labels = get_features_labels(df_train)

test_features, test_labels = get_features_labels(df_test)
print( train_features.shape )

print( train_labels.shape )
def examine_data( index ):

    plt.figure()

    _ = plt.imshow(np.reshape(train_features[index,:], (28,28)), 'gray')
examine_data(42)
# Keras has a utility to one-hot encode categorical data

train_labels = tf.keras.utils.to_categorical(train_labels)

test_labels = tf.keras.utils.to_categorical(test_labels)
train_labels.shape
test_labels[42]
#define the model

model = tf.keras.Sequential()



# add a dense (fully-connected) layer that takes in the 784 inputs of the 28x28 image

model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(784,)))



#add a dense layer that takes the output of the previous layer and has 20 output nodes

model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))



#add a dense layer that takes the output of the previous layer, and has 10 output nodes and utilizes the softmax function to 

# generate the probability of 1 of 10 classes

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



model.compile(loss='categorical_crossentropy',

             optimizer='rmsprop',

             metrics=['accuracy'])



model.summary()
EPOCHS=6

BATCH_SIZE=128
model.fit(train_features, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
test_loss, test_acc = model.evaluate(test_features, test_labels)
print('test_accuracy:', test_acc)