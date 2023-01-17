import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
DATA_DIR = '/kaggle/input/cars196'

[train_ds, test_ds], ds_info = tfds.load(
    "cars196",
    # Reserve 10% for validation and 50% for test
    split=["train", "test"],
    as_supervised=True,  # Include labels
    with_info=True,
    download=False,
    data_dir=DATA_DIR,
)
tfds.visualization.show_examples(train_ds, ds_info)
height, width = 150, 150


batch_size = 32

def augment_func(image,label):
  # augment data
  return image, label


base_model = tf.keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(height, width, 3),
    include_top=False, # Do not include the final ImageNet classifier layer at the top.
)  

base_model.trainable = True # We want to update all the model weights, so set this to true.

# Create new model on surrounding our pretrained base model.
inputs = tf.keras.Input(shape=(height, width, 3))

# Pre-trained Xception weights requires that input be normalized
# from (0, 255) to a range (-1., +1.), the normalization layer
# does the following, outputs = (inputs - mean) / sqrt(var)
norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 3)
var = mean ** 2
# Scale inputs to [-1, +1]
x = norm_layer(inputs)
norm_layer.set_weights([mean, var])

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x) # this is a neural network operation to help adapt the features learned by the pretrained model to our specific task.
x = keras.layers.Dropout(0.5)(x)  # Regularize with dropout
num_outputs = ds_info.features['label'].num_classes # This is the number of output variables we want, 196 in this case.
outputs = keras.layers.Dense(num_outputs, activation="softmax")(x) # Use activation=softmax for classification, and activation=None for regression.
model = keras.Model(inputs, outputs)

model.summary()

model.save("model.h5", save_format="h5")