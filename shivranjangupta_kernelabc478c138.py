# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from tensorflow.keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

import tensorflow as tf

train_data = pd.read_csv("../input/digit-recognizer/train.csv")

test_data = pd.read_csv("../input/digit-recognizer/test.csv")

train_data.head()
X = np.array(train_data.drop("label", axis=1)).astype('float32')

y = np.array(train_data['label']).astype('float32')

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X[i].reshape(28, 28), cmap=plt.cm.binary)

    plt.xlabel(y[i])

plt.show()



X = X / 255.0

X = X.reshape(-1, 28, 28, 1)



y = to_categorical(y)



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_test = np.array(test_data).astype('float32')

X_test = X_test / 255.0

X_test = X_test.reshape(-1, 28, 28, 1)

plt.figure(figsize=(10,10))

test_data = test_data.values.reshape(-1,28,28,1)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

model.summary()

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model1.png')
#increse to epochs to 30 for better accuracy

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

#history = model.fit(X_train, y_train, epochs=12, batch_size=85, validation_data=(X_val, y_val))
# from keras.optimizers import RMSprop



# # Define the optimizer

# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 21 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 85
# Fit the model

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
# With data augmentation to prevent overfitting (accuracy 0.99286)



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)

accuracy = history.history['accuracy']

val_accuracy = history.history['val_accuracy']

epochs = range(len(accuracy))



print(model.evaluate(X_val, y_val))


# Predict the values from the validation dataset

y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

# confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# # plot the confusion matrix

# plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# predict results

results = model.predict(test_data, verbose=0)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)