import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
# View Class Balance

label = train['label']

classes = sb.countplot(label)

classes.set_title('Class Balance')

classes.set_xlabel("Digit")

classes.set_ylabel("Count")

plt.show()



# Check for missing data

print(train.isnull().sum().max()) # No column is missing data

print(test.isnull().sum().max())
data = train.drop(labels = 'label', axis = 1) # Drop labels for data formatting



# Standardize the data

test = test / 255.0

data = data / 255.0 



# Reshape Data from 1x756x1 to 28x28x1

data = data.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
plt.imshow(data[450][:,:,0], cmap = 'gray_r')
# Change label to categorical

label = to_categorical(label, num_classes = 10)



# Split training into test and train set

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size = .15, random_state = 4)
# Now I want to train a CNN on the X_train dataset

cnn = Sequential()



cnn.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu',

              input_shape = (28, 28, 1)))

cnn.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

cnn.add(MaxPool2D(pool_size = (2,2)))

cnn.add(Dropout(0.25))



cnn.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

cnn.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

cnn.add(MaxPool2D(pool_size = (2,2)))

cnn.add(Dropout(0.25))



cnn.add(Flatten())

cnn.add(Dense(256, activation = "relu"))

cnn.add(Dropout(0.50))

cnn.add(Dense(10, activation = "softmax"))



# Define Optimizer (could also use Stochastic Gradient Descent)

optimizer = RMSprop(lr=.001,rho=0.9,epsilon=1e-8,decay=0.0)



# Compile the model

cnn.compile(optimizer="sgd", loss = "categorical_crossentropy", 

           metrics=["accuracy"])



epochs = 5

batch_size = 64



# Data Generator / Data Augmentation

gen = ImageDataGenerator(rotation_range=10,

        zoom_range = 0.1,

        width_shift_range=0.1,

        height_shift_range=0.1)



gen.fit(X_train)



cnn.fit_generator(gen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,Y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size)
Y_pred = cnn.predict(X_test)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

Y_true = np.argmax(Y_test, axis = 1)

confusion_matrix(Y_true, Y_pred_classes)
