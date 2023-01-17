# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# create the training & test sets

train = pd.read_csv("../input/digit-recognizer/train.csv")

test= pd.read_csv("../input/digit-recognizer/test.csv")



print("train.shape:", train.shape)

print("test.shape:", test.shape)
X_train = train.iloc[:,1:].values.astype('float32') # all pixel values

Y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')
#expand 1 more dimention as 1 for colour channel gray

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)



X_train, X_test = X_train / 255.0, X_test / 255.0
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 2)
print("X_train.shape:", X_train.shape)

print("X_val.shape:", X_val.shape)

print("Y_train.shape:", Y_train.shape)

print("Y_val.shape:", Y_val.shape)

print("X_test.shape:", X_test.shape)
# In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()

model.add(Conv2D(32, (5,5), activation = 'relu', input_shape = (28,28,1)))

model.add(Conv2D(32, (5,5), activation = 'relu'))

model.add(MaxPool2D((2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPool2D((2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.summary()
model.compile(optimizer = 'adam' , loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy', 

                                            patience = 3, 

                                            verbose = 1, 

                                            factor = 0.5, 

                                            min_lr = 0.00001)

epochs = 30

batch_size = 86
# With data augmentation to prevent overfitting

datagen = ImageDataGenerator(

        rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip = False,  # randomly flip images

        vertical_flip = False)  # randomly flip images



# compute quantities required for featurewise normalization

# (std, mean, and principal components if ZCA whitening is applied)

datagen.fit(X_train)
# fits the model on batches with real-time data augmentation

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size = batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, 

                              steps_per_epoch = X_train.shape[0] // batch_size,

                              callbacks = [learning_rate_reduction])
#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, 'r')

plt.plot(epochs, val_acc, 'b')

plt.title('Training and validation accuracy')

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend(["Accuracy", "Validation Accuracy"])



plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r')

plt.plot(epochs, val_loss, 'b')

plt.title('Training and validation loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend(["Loss", "Validation Loss"])



plt.figure()
predictions = model.predict_classes(X_test, verbose = 0)

submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)), "Label": predictions})

submissions.to_csv("cnn_output.csv", index = False, header = True)