%%capture

!pip install scipy==1.1.0

!pip install keras-vis
import numpy as np

import keras



from keras.datasets import mnist

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, Input

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



batch_size = 128

num_classes = 10

epochs = 1



# Input Image Dimensions

img_rows, img_cols = 28, 28



(x_train, y_train), (x_test, y_test) = mnist.load_data()



if K.image_data_format() == "channels_first":

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)

    

x_train = x_train.astype("float32")

x_test = x_test.astype("float32")

x_train /= 255

x_test /= 255

print("Training Shape:", x_train.shape)

print("Training Samples:", x_train.shape[0])

print("Testing Samples:", x_test.shape[0])



# Convert Class Vectors to Binary Class Vectors

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=input_shape))

model.add(Conv2D(64, (3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation="softmax", name="preds"))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=["accuracy"])



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))



score = model.evaluate(x_test, y_test, verbose=0)

print("Test Loss:", score[0])

print("Test Accuracy:", score[1])
from vis.visualization import visualize_activation

from vis.utils import utils

from keras import activations

from matplotlib import pyplot as plt

%matplotlib inline



plt.rcParams["figure.figsize"] = (18,6)



# Search for the layer index by name

layer_idx = utils.find_layer_idx(model, "preds")



# Swap softmax activation with linear activation

model.layers[layer_idx].activation = activations.linear

model = utils.apply_modifications(model)



# Output node used for maximization

# First input prediction of a 0

filter_idx = 0

img = visualize_activation(model, layer_idx, filter_indices=filter_idx)

plt.imshow(img[..., 0])
img = visualize_activation(model,

                           layer_idx,

                           filter_indices=filter_idx,

                           input_range=(0., 1.),

                           verbose=True)

plt.imshow(img[..., 0])
# Generate visualizations for all classes (0-9)

for output_idx in np.arange(10):

    img = visualize_activation(model, layer_idx, filter_indices=output_idx, input_range=(0., 1.))

    plt.figure()

    plt.title(f"Network Perception of {output_idx}")

    plt.imshow(img[..., 0])