import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
x_data = np.load('/kaggle/input/sign-language-digits-dataset/X.npy')

y_data = np.load('/kaggle/input/sign-language-digits-dataset/Y.npy')



print('Shape of x data: ', x_data.shape)

print('Shape of y data: ', y_data.shape)
plt.figure(figsize=(10,10))

plt.subplot(1,3,1)

plt.imshow(x_data[0], cmap='gray')

plt.subplot(1,3,2)

plt.imshow(x_data[100], cmap='gray')

plt.title('Example Images')

plt.subplot(1,3,3)

plt.imshow(x_data[500], cmap='gray');
# numpy reshape function is used to change matrix format:

x_data = x_data.reshape(-1, 64, 64, 1)

print('New shape of x_data: ', x_data.shape)



# y_data is already in proper matrix format:

print('Shape of y_data: ', y_data.shape)
# Using 80% of the data for training and 20% for testing. 



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state = 1)



print('Shape of x_train: ', x_train.shape)

print('Shape of y_train: ', y_train.shape)

print('....')

print('Shape of x_test: ', x_test.shape)

print('Shape of y_test: ', y_test.shape)
from keras.preprocessing.image import ImageDataGenerator



train_gen = ImageDataGenerator(

            rotation_range = 5,        # 5 degrees of rotation will be applied

            zoom_range = 0.1,          # 10% of zoom will be applied

            width_shift_range = 0.1,   # 10% of shifting will be applied

            height_shift_range = 0.1)  # 10% of shifting will be applied



train_gen.fit(x_train)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D



# Creating model structure

model = Sequential()

# Adding the first layer of CNN

model.add(Conv2D(filters=20, kernel_size=(4,4), padding='Same', activation='relu', input_shape=(64, 64, 1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.15))

# Adding the second layer of CNN

model.add(Conv2D(filters=30, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.15))

# Flattening the x_train data

model.add(Flatten()) 

# Creating fully connected NN with 4 hidden layers

model.add(Dense(220, activation='relu'))

model.add(Dropout(0.15))

model.add(Dense(150, activation='relu'))

model.add(Dropout(0.15))

model.add(Dense(80, activation='relu'))

model.add(Dropout(0.15))

model.add(Dense(10, activation='softmax'))
# Defining the optimizer



from keras.optimizers import Adam



optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.99)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
batch_size = 100

epochs = 25
history = model.fit_generator(train_gen.flow(x_train, y_train, batch_size = batch_size), 

                                                  epochs = epochs, 

                                                  validation_data = (x_test, y_test), 

                                                  steps_per_epoch = x_train.shape[0] // batch_size)
# Visiualize the validation loss and validation accuracy progress:



plt.figure(figsize=(13,5))

plt.subplot(1,2,1)

plt.plot(history.history['val_loss'], color = 'r', label = 'validation loss')

plt.title('Validation Loss Function Progress')

plt.xlabel('Number Of Epochs')

plt.ylabel('Loss Function Value')



plt.subplot(1,2,2)

plt.plot(history.history['val_accuracy'], color = 'g', label = 'validation accuracy')

plt.title('Validation Accuracy Progress')

plt.xlabel('Number Of Epochs')

plt.ylabel('Accuracy Value')

plt.show()
# Confusion Matrix



from sklearn.metrics import confusion_matrix

import seaborn as sns



# First of all predict labels from x_test data set and trained model

y_pred = model.predict(x_test)



# Convert prediction classes to one hot vectors

y_pred_classes = np.argmax(y_pred, axis = 1)



# Convert validation observations to one hot vectors

y_true_classes = np.argmax(y_test, axis = 1)



# Create the confusion matrix

confmx = confusion_matrix(y_true_classes, y_pred_classes)

f, ax = plt.subplots(figsize = (8,8))

sns.heatmap(confmx, annot=True, fmt='.1f', ax = ax)

plt.xlabel('Predicted Labels')

plt.ylabel('True Labels')

plt.title('Confusion Matrix')

plt.show();