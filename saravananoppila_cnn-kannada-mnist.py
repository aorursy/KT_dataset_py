# Import required packages #

import matplotlib.pyplot as plt,seaborn as sns,pandas as pd,numpy as np

from keras.models import Sequential, load_model

from keras.layers.core import Dense, Dropout, Activation

from keras.layers import Conv2D, MaxPooling2D,MaxPool2D,Flatten,BatchNormalization

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import Adam

import tensorflow as tf

from sklearn.model_selection import train_test_split
# Read data

train_data = pd.read_csv('../input/Kannada-MNIST/train.csv')
# Drop 'label' column

x_train = train_data.drop(labels = ["label"],axis = 1) 

y_train = train_data["label"]
# Plot target class #

sns.countplot(train_data.label)
x_train = x_train.values.astype('float32') / 255
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state=42)
# let's print the shape before we reshape and normalize

print("X_train shape", x_train.shape)

print("y_train shape", y_train.shape)

print("X_test shape", x_test.shape)

print("y_test shape", y_test.shape)
# building the input vector from the 28x28 pixels

X_train = x_train.reshape(-1, 28, 28,1)

X_test = x_test.reshape(-1, 28, 28,1)

# print the final input shape ready for training

print("Train matrix shape", X_train.shape)

print("Test matrix shape", X_test.shape)
# one-hot encoding using keras' numpy-related utilities

n_classes = 10

print("Shape before one-hot encoding: ", y_train.shape)

Y_train = np_utils.to_categorical(y_train, n_classes)

Y_test = np_utils.to_categorical(y_test, n_classes)

print("Shape after one-hot encoding: ", Y_train.shape)
# Build LeNet-5 Convolution neural network #

model = Sequential()

model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(128, kernel_size = 4, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')



adam = Adam(lr=5e-4)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
model.summary()
# Set a learning rate annealer

reduce_lr = ReduceLROnPlateau(monitor='val_acc', 

                                patience=3, 

                                verbose=1, 

                                factor=0.2, 

                                min_lr=1e-6)
# Data Augmentation

datagen = ImageDataGenerator(

            rotation_range=10, 

            width_shift_range=0.1, 

            height_shift_range=0.1, 

            zoom_range=0.1)

history = datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=100), steps_per_epoch=len(X_train)/100, 

                    epochs=20, validation_data=(X_test, Y_test), callbacks=[reduce_lr])
# Evaluate the model with test data

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])

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
# Load test data #

test_data = pd.read_csv('../input/Kannada-MNIST/test.csv')

id_ = test_data.id 

test_data = test_data.drop("id",axis="columns")

test_data = test_data.values.reshape(-1, 28, 28,1)

test_data = test_data.astype('float32')

# Normalise test data #

test_data /= 255

print("Test data matrix shape", test_data.shape)
# predict test data #

y_pred = model.predict_classes(test_data, verbose=0)

print(y_pred)
# Predict indivdual input image #

i = 7

predicted_value = np.argmax(model.predict(X_test[i].reshape(1,28, 28,1)))

print('predicted value:',predicted_value)

plt.imshow(X_test[i].reshape([28, 28]), cmap='Greys_r')
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

sample_sub['label']=y_pred

sample_sub.to_csv('submission.csv',index=False)