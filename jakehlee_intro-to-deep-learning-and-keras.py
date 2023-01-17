import numpy as np

import keras

from keras.datasets import mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data()



# This is a quirk of keras; since the images are grayscale,

# we need to add an axis so the shape is (60000, 28, 28, 1)

# instead of (60000, 28, 28)



x_train = x_train[:,:,:,np.newaxis]

x_test = x_test[:,:,:,np.newaxis]



# We're also going to convert 0~255 to 0~1 float.

x_train = x_train.astype(np.float)

x_test = x_test.astype(np.float)

x_train /= 255

x_test /= 255



# Finally, the classes need to be one-hot encoded.

# That is:

# 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0]

# 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0]

# etc.

# This is to match what the network will output - 

# there are 10 nodes at the end, each with its own

# confidence of its class. The ground truth should be

# 100% confidence of the true label.



y_train = keras.utils.to_categorical(y_train, 10)

y_test = keras.utils.to_categorical(y_test, 10)
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

#                        Remember these?



# By the way, we really like powers of 2 for the number

# of nodes at each layer.

model = Sequential([

    # input layer, 16 conv (spatial) perceptrons of size (3,3)

    # image shape is (28, 28, 1). If it was color it'd be (28, 28, 3)

    Conv2D(8, (3,3), activation='relu', input_shape=(28, 28, 1)),

    # Now for the max pooling to make the size smaller

    MaxPooling2D(pool_size=(2,2)),

    # Flatten before sending to Dense (2D to 1D)

    Flatten(),

    # Output layer with 10 nodes for 10 classes, with softmax

    Dense(10, activation='softmax')

])
model.compile(loss=keras.losses.categorical_crossentropy,

             optimizer=keras.optimizers.SGD(),

             metrics=['accuracy'])
import time

start = time.time()

model.fit(x_train,        # training data

          y_train,        # training labels

          batch_size=16,  # how many training examples you want to give at once

          verbose=1,      # print progress in console

          validation_data=(x_test, y_test),  # validation data to check generalization

          epochs=5)       # how many times to go through the entire training set

end = time.time()

print("Training took", end-start, "seconds.")