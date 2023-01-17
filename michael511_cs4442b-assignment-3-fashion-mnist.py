# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os



from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical



# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Load both train and test data set

train = pd.read_csv("../input/fashion-mnist_train.csv") # Save training data to var train

test = pd.read_csv("../input/fashion-mnist_test.csv") # Save testing data to var test



# let's look at first five train samples

train.head(5)
Y_train = train.label

X_train = train.drop(["label"], axis=1)

X_test = test.drop(["label"], axis=1)

Y_test = test.label



plt.figure(figsize=(18, 8))

sns.countplot(Y_train, palette="deep")

plt.title("Number of Classes")

plt.show()
# Normalization

X_train = X_train / 255.0

X_test = X_test / 255.0



# Format Data as 3D Matrix for use with Keras

X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)



#Label Encoding

Y_train = to_categorical(Y_train, num_classes=10)
from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.3, random_state = 42)



print("x_train shape",x_train.shape)

print("x_test shape",x_val.shape)

print("y_train shape",y_train.shape)

print("y_test shape",y_val.shape)
plt.figure(figsize=(18, 8))



for i in range(5):

    plt.subplot(2, 5, i+1)

    img = train[train.label==i].iloc[0, 1:].values

    img = img.reshape((28, 28))

    plt.imshow(img, cmap='gray')

    plt.title("Class " + str(i))

    plt.axis('off')

    

plt.show()
# Defines the options used in our image augmentation

datagen = ImageDataGenerator(vertical_flip=True, # Some images may be flipped vertically

                             horizontal_flip=False,

                             height_shift_range=0.1, # Some images may be shifted vertically

                             width_shift_range=0.1, # Some images may be shifted horizontally

                             rotation_range=30, # Some images may be augmented through rotation, up to 45 degrees

                             zoom_range=0.1) # Some images may be augmented with a slight zoom



#Fit params from data

datagen.fit(x_train)



#Configure and retrieve augmented images for comparison

for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):

    for i in range(0, 9):

        plt.subplot(330 + 1 + i)

        plt.imshow(X_batch[i].reshape(28, 28), cmap='gray')

        plt.axis('off')

    # show the plot

    plt.show()

    break
CNNmodel = Sequential()



# Conv + Maxpool

CNNmodel.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation='relu', input_shape=(28,28,1)))

CNNmodel.add(MaxPool2D(pool_size = (2,2)))



# Dropout

CNNmodel.add(Dropout(0.25))



# Conv + Maxpool

CNNmodel.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

CNNmodel.add(MaxPool2D(pool_size=(2,2)))



# Dropout 

CNNmodel.add(Dropout(0.25))



# Conv + Maxpool

CNNmodel.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

CNNmodel.add(MaxPool2D(pool_size=(2,2)))



# Dropout

CNNmodel.add(Dropout(0.25))



# Flatten 3D Feature Vector into 1D array

CNNmodel.add(Flatten())



# Fully Connected Layer

CNNmodel.add(Dense(256, activation='relu'))

CNNmodel.add(Dropout(0.25))

CNNmodel.add(Dense(256, activation='relu'))

CNNmodel.add(Dropout(0.1))

CNNmodel.add(Dense(10, activation='softmax'))



#Adam Optimizer

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# Compile the Model

CNNmodel.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
# Define Epoch and Batch Size 

# 'Epoch' refers to the number of times the algorithm will see each data set. 'Batch' refers to the size of each individual Epoch



epochs = 50

batchSize = 300
CNN = CNNmodel.fit_generator(datagen.flow(x_train, y_train, batch_size=batchSize), epochs=epochs, validation_data=(x_val, y_val), steps_per_epoch=x_train.shape[0] // batchSize)
print("Accuracy after fitting: {:.2f}%".format(CNN.history['acc'][-1]*100))
plt.figure(figsize=(18,6))



plt.subplot(1,2,1)

plt.plot(CNN.history['loss'], color="black", label = "Loss")

plt.plot(CNN.history['val_loss'], color="red", label = "Validation Loss")

plt.ylabel("Loss")

plt.xlabel("Number of Epochs")

plt.legend()



plt.subplot(1,2,2)

plt.plot(CNN.history['acc'], color="black", label = "Accuracy")

plt.plot(CNN.history['val_acc'], color="red", label = "Validation Accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Number of Epochs")

plt.legend()

plt.show()
Y_test = to_categorical(Y_test, num_classes=10) # One-Hot Encoding
score = CNNmodel.evaluate(X_test, Y_test)

print("Test Accuracy: {:.2f}%".format(score[1]*100))

print("Test Loss: {:.3f}".format(score[0]))
Y_pred = CNNmodel.predict(X_test)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

Y_true = np.argmax(Y_test, axis = 1)

confusionMatrix = confusion_matrix(Y_true, Y_pred_classes)



f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(confusionMatrix, annot=True, linewidths=0.1, cmap = "gist_yarg_r", linecolor="black", fmt='.0f', ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()



# For loop to print how many items of each class have been incorrectly estimated

for i in range(len(confusionMatrix)):

    print("Class:",str(i))

    print("Number of Wrong Prediction:", str(sum(confusionMatrix[i])-confusionMatrix[i][i]), "out of 1000")