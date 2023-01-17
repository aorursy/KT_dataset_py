# Install Emnist packages to load data #

!pip install emnist
# Import required packages #

from keras.models import Sequential, load_model

from keras.layers.core import Dense, Dropout, Activation

from keras.utils import np_utils

from emnist import extract_training_samples

from emnist import extract_test_samples

import numpy as np,seaborn as sns,matplotlib.pyplot as plt
# extract train and test data #

x_train, y_train = extract_training_samples('letters')

x_test, y_test = extract_test_samples('letters')
# Plot target class #

sns.countplot(y_train)
# Plot train input images #

fig = plt.figure()

for i in range(9):

  plt.subplot(3,3,i+1)

  plt.tight_layout()

  plt.imshow(x_train[i], cmap='gray', interpolation='none')

  plt.title("Digit: {}".format(y_train[i]))

  plt.xticks([])

  plt.yticks([])
# let's print the shape before we reshape and normalize #

print("X_train shape", x_train.shape)

print("y_train shape", y_train.shape)

print("X_test shape", x_test.shape)

print("y_test shape", y_test.shape)
# building the input vector from the 28x28 pixels #

X_train = x_train.reshape(124800, 784)

X_test = x_test.reshape(20800, 784)

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')
# normalizing the data to help with the training #

X_train /= 255

X_test /= 255



# print the final input shape ready for training

print("Train matrix shape", X_train.shape)

print("Test matrix shape", X_test.shape)
# one-hot encoding using keras' numpy-related utilities #

n_classes = 27

print("Shape before one-hot encoding: ", y_train.shape)

Y_train = np_utils.to_categorical(y_train, n_classes)

Y_test = np_utils.to_categorical(y_test, n_classes)

print("Shape after one-hot encoding: ", Y_train.shape)

# Build Fully connected neural network with 4 layer #

model = Sequential()

model.add(Dense(512, input_shape=(784,),activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(27,activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# Fit the model #

history = model.fit(X_train, Y_train,

          batch_size=128, epochs=50,

          verbose=2,

          validation_split=0.1)
# Evaluate the model with test data

score = model.evaluate(X_test, Y_test, verbose=1)

print("Test score:", score[0])

print('Test accuracy:', score[1])
# Plot Accuracy and Loss graph #

f = plt.figure(figsize=(20,7))

f.add_subplot(121)

plt.plot(history.epoch,history.history['accuracy'],label = "accuracy")

plt.plot(history.epoch,history.history['val_accuracy'],label = "val_accuracy")

plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()





f.add_subplot(122)

plt.plot(history.epoch,history.history['loss'],label="loss") 

plt.plot(history.epoch,history.history['val_loss'],label="val_loss")

plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()
# Predict indivdual input image #

i = 9713

predicted_value = np.argmax(model.predict(X_test[i].reshape(1,784)))

print('predicted value:',predicted_value)

plt.imshow(X_test[i].reshape([28, 28]), cmap='Greys_r')
