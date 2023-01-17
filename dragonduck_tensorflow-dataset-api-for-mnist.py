import os

import tensorflow as tf

tf.enable_eager_execution()

import pandas as pd

import numpy as np

import sklearn.model_selection

import sklearn.preprocessing
# Parameters

image_dims = (28, 28, 1)

batchsize = 64

num_labels = 10
data_training, data_validation = sklearn.model_selection.train_test_split(

    pd.read_csv("../input/train.csv"), test_size=0.2)

data_test = pd.read_csv("../input/test.csv")



X_training = data_training.drop("label", axis=1).values

X_validation = data_validation.drop("label", axis=1).values

X_test = data_test.values.reshape((-1,) + image_dims)



y_training = data_training["label"]

y_validation = data_validation["label"]
def preprocess_training_data(image, label):

    """

    Converts 1D array into 2D matrix and normalized entries

    """

    image = tf.reshape(image, image_dims)

    image = tf.dtypes.cast(image, tf.float64)

    min_val = tf.reduce_min(image)

    max_val = tf.reduce_max(image)

    image = (image - min_val) / (max_val - min_val)

    label = tf.one_hot(indices=label, depth=num_labels)

    return image, label



def preprocess_testing_data(image):

    """

    Converts 1D array into 2D matrix and normalized entries

    """

    image = tf.reshape(image, image_dims)

    image = tf.dtypes.cast(image, tf.float64)

    min_val = tf.reduce_min(image)

    max_val = tf.reduce_max(image)

    image = (image - min_val) / (max_val - min_val)

    return image
def make_dataset(data, labels=None):

    if labels is None:

        dataset = tf.data.Dataset.from_tensor_slices(data)

        dataset = dataset.map(preprocess_testing_data, num_parallel_calls=-1)

    else:

        dataset = tf.data.Dataset.from_tensor_slices((data, labels))

        dataset = dataset.shuffle(buffer_size=len(data))

        dataset = dataset.map(preprocess_training_data, num_parallel_calls=-1)

    dataset = dataset.batch(batchsize)

    dataset = dataset.prefetch(1)

    return dataset
dataset_training = make_dataset(X_training, y_training)

dataset_validation = make_dataset(X_validation, y_validation)

dataset_testing = make_dataset(X_test)
def make_model():

    input_layer = tf.keras.layers.Input(

        shape=image_dims, name="Input")

    layer = tf.keras.layers.Conv2D(

        filters=64, kernel_size=(3, 3), 

        activation="relu", use_bias=True,

        kernel_initializer="glorot_uniform", 

        bias_initializer="glorot_uniform")(input_layer)

    layer = tf.keras.layers.Dropout(rate=0.3)(layer)

    layer = tf.keras.layers.Conv2D(

        filters=64, kernel_size=(3, 3), 

        activation="relu", use_bias=True,

        kernel_initializer="glorot_uniform", 

        bias_initializer="glorot_uniform")(layer)

    layer = tf.keras.layers.Dropout(rate=0.3)(layer)

    layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(layer)

    layer = tf.keras.layers.Conv2D(

        filters=128, kernel_size=(3, 3), 

        activation="relu", use_bias=True,

        kernel_initializer="glorot_uniform", 

        bias_initializer="glorot_uniform")(layer)

    layer = tf.keras.layers.Dropout(rate=0.3)(layer)

    layer = tf.keras.layers.Conv2D(

        filters=128, kernel_size=(3, 3), 

        activation="relu", use_bias=True,

        kernel_initializer="glorot_uniform", 

        bias_initializer="glorot_uniform")(layer)

    layer = tf.keras.layers.Dropout(rate=0.3)(layer)

    layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(layer)

    layer = tf.keras.layers.Flatten()(layer)

    output_layer = tf.keras.layers.Dense(units=num_labels, activation="softmax")(layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer="adam", 

        loss="categorical_crossentropy",

        metrics=["accuracy"])

    return model
model = make_model()
model.summary()
model.fit(

    dataset_training, epochs=5, validation_data=dataset_validation, 

    steps_per_epoch=int(len(data_training)/batchsize), 

    validation_steps=int(len(data_validation)/batchsize))
pred = model.predict(dataset_testing)
output = pd.DataFrame(

    data={

        "ImageId": data_test.index+1, 

        "Label": np.argmax(pred, axis=1)})



output.head()

output.to_csv("TestPrediction.csv", index=False)