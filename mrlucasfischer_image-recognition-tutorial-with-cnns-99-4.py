import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D

from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ReduceLROnPlateau



# Loading the data

# I'm adding the ".values" to the test set to convert it to a numpy array

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv").values



print(train.shape, test.shape)
# I'm adding the ".values" to convert them to numpy arrays for easier algebraic manipulation

train_y = train["label"].values

train_x = train.drop(labels = ["label"], axis = 1).values
train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))

test = test.reshape((test.shape[0], 28, 28, 1))
train_x = train_x.astype("float32") / 255.0

test = test.astype("float32") / 255.0
# We one-hot encode the trainning labels

train_y = keras.utils.to_categorical(train_y, 10)
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, test_size = 0.1)
# We initialize the Keras model as a Sequential model

model = Sequential()



# The input layer is a convolutional layer with 32 filters

# The shape of the kernel in this layer is 3x3

# We add padding in this layer (so we can start the kernel right at the beginning of the image)

# and in this case we use padding "same" for it to add values to the padding that are copied from the original matrix (it could also be 0)

model.add(Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1)))



# For this layer we add a ReLU activation

# We need to add ReLU because a convolution is still a linear transformation

# so we add ReLU for it to be a non linear transformation

model.add(Activation("relu"))



# We add batch normalization here

# This normalizes the output from the previous layer in order

# for the input of the next layer to be normalized

# In this case we put the channels at the end so we don't need to specify the axis of normalization

# otherwise we would need to specify

model.add(BatchNormalization())



model.add(Conv2D(32, (3, 3), padding="same", activation = "relu", input_shape=(28, 28, 1)))

model.add(BatchNormalization())



# In this layer we Pool the layer before in order to reduce the number of features

# Since we are using a 2x2 pooling size we are keeping only half of the features in each dimension

# So instead of a 28*28 vector we now have a 14*14 tensor

# Since we are omitting the stride Keras assumes the same stride as pool size which is what we want

model.add(MaxPooling2D(pool_size=(2, 2)))



# We add a dropout layer of 25% dropout for regularization

model.add(Dropout(0.25))



# We add another convolution layer, in this case we don't need to specify the input shape

# because keras finds out the right input shape

model.add(Conv2D(64, (3, 3), padding="same", activation = "relu"))

model.add(BatchNormalization())



model.add(Conv2D(64, (3, 3), padding="same", activation = "relu"))

model.add(BatchNormalization())



model.add(Conv2D(64, (3, 3), padding="same", activation = "relu"))

model.add(BatchNormalization())



# After this pooling we have a 7*7 tensor

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
# We add a Flatten layer in order to transform the input tensor into a vector

# In this case we had a 7*7*64 (7*7*the number of filters we have)

model.add(Flatten())



# We have 512 neurons in this layer

model.add(Dense(512, activation = "relu"))



model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
# defining the learning rate, the number of epochs and the batch size

INIT_LR = 0.001

NUM_EPOCHS = 30

BS = 86

opt = RMSprop(lr = INIT_LR, rho=0.9, epsilon=1e-08, decay=0.0)



# We track the metrics "accuracy"

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



# Reduce the learning rate by half if validation accuracy has not increased in the last 3 epochs

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)



fitted_network = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=BS, epochs=NUM_EPOCHS, callbacks=[learning_rate_reduction])
# predict results

results = model.predict(test)



# now we want to retrieve the index that had the higher probability, that will be our prediction

results = np.argmax(results,axis = 1)



results = pd.Series(results, name = "Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist.csv",index = False)