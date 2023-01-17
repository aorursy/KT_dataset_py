import pandas as pd

import numpy as np



import tensorflow as tf



tf.test.gpu_device_name()
train_df = pd.read_csv("../input/Kannada-MNIST/train.csv")

test_df = pd.read_csv("../input/Kannada-MNIST/test.csv")





features = train_df.iloc[:,1:].values

targets = train_df.iloc[:,0].values

validation = test_df.iloc[:,1:].values



print("training images shape {}".format(features.shape))

print("training labels shape {}".format(targets.shape))

print("validation images shape {}".format(validation.shape))
# reshape images



flattened_features = features.reshape(features.shape[0], 28, 28, 1)

flattened_validation = validation.reshape(validation.shape[0], 28, 28, 1)



print("x_flattened size {}".format(flattened_features.shape))

print("val_flattened size {}".format(flattened_validation.shape))
def normalize_data(df):

  df = df / 255.

  return df



normalized_features = normalize_data(flattened_features)

normalized_validation = normalize_data(flattened_validation)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



# test train split



X_train, X_test, y_train, y_test = train_test_split(normalized_features, targets, test_size=0.2, random_state=7)



# viz

plt.imshow(X_train[0][:,:,0])
# Build CNN with Keras

# standard, with the addition of swish!



# Config

EPOCHS = 250

BATCH_SIZE = 256

LEARNING_RATE = 0.001 # 0.002



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from tensorflow.keras import backend as k



# add custom swish activation function

from tensorflow.keras.backend import sigmoid

def swish(x, beta = 1):

    return (x * sigmoid(beta * x))



from tensorflow.keras.layers import Activation

tf.keras.utils.get_custom_objects().update({'swish': Activation(swish)})



# Build model



model = Sequential()



# add layers

model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same', activation='swish', input_shape=(28, 28, 1)))

model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same', activation='swish', input_shape=(28, 28, 1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='Same', activation='swish'))

model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='Same', activation='swish'))

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='swish'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))



## optimizer

keras_optimizer = tf.keras.optimizers.Nadam(lr=LEARNING_RATE)



# Compile model

model.compile(optimizer=keras_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
trained_model = model.fit(x=X_train, y=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")



class_predictions = model.predict_classes(normalized_validation)



sample_submission['label'] = pd.Series(class_predictions)



sample_submission.to_csv("submission.csv",index=False)