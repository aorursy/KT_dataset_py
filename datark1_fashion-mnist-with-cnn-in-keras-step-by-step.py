import numpy as np                                   # linear algebra

import pandas as pd                                  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt                      # library used for plotting data

from sklearn.model_selection import train_test_split # method used for splitting data set into trining and testing sets

import warnings                                      # libraries to deal with warnings

warnings.filterwarnings("ignore")



print("numpy version:", np.__version__)

print("pandas version:", pd.__version__)
raw_data_train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

raw_data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
raw_data_train.head()
raw_data_train.dtypes.unique()
subset_1 = raw_data_train.iloc[:1000,1:]

plt.subplots(figsize=(10,5))

plt.hist(subset_1, bins=256, fc='k', ec='k',histtype='step')

plt.show()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



plt.figure(figsize=(12,9))

for i in range(0,12):

    plt.subplot(3,4,i+1)

    image_resized = np.resize(raw_data_train.iloc[i,1:].values,(28,28))

    plt.title(class_names[raw_data_train.iloc[i,0]])

    plt.imshow(image_resized, cmap='gray', interpolation='none')

    plt.axis('off')
X = np.array(raw_data_train.iloc[:, 1:])

y = pd.get_dummies(np.array(raw_data_train.iloc[:, 0]))



# alternative:

#from keras.utils import to_categorical

#y = to_categorical(np.array(raw_data_train.iloc[:, 0]))
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=12)
im_rows, im_cols = 28, 28

input_shape = (im_rows, im_cols, 1)



# Test data

X_test = np.array(raw_data_test.iloc[:, 1:])

y_test = pd.get_dummies(np.array(raw_data_test.iloc[:, 0]))



# train and validate sets

X_train = X_train.reshape(X_train.shape[0], im_rows, im_cols, 1)

X_validate = X_validate.reshape(X_validate.shape[0], im_rows, im_cols, 1)

X_test = X_test.reshape(X_test.shape[0], im_rows, im_cols, 1)



# normalisation

X_train = X_train/255

X_validate = X_validate/255

X_test = X_test/255



print("X_train shape:", X_train.shape)

print("X_validate shape:", X_validate.shape)

print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)

print("y_validate shape:", y_validate.shape)

print("y_test shape:", y_test.shape)
import keras # main keras package

from keras.models import Sequential # sequential model

from keras.layers import Dropout, Flatten, MaxPooling2D # layers with layers operations

from keras.layers import Dense,Conv2D  # layers types



print("Keras version:", keras.__version__)
num_classes = 10



#defining model

model = Sequential()



# the first layer - convolutional layer

model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding="valid",

                 input_shape=input_shape))

model.add(MaxPooling2D(2))

model.add(Dropout(0.3))



# the second layer - convolutional layer

model.add(Conv2D(128, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='valid'))

model.add(MaxPooling2D(2))

model.add(Dropout(0.3))



# the third layer - convolutional layer

model.add(Conv2D(256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))

model.add(Dropout(0.3))

model.add(Flatten())



# the fourth layer - dense layer

model.add(Dense(256, kernel_initializer='he_normal', activation='relu'))



# the fifth layer - dense layer

model.add(Dense(512, kernel_initializer='he_normal', activation='relu'))



# The last layer - dense layer

model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer="adam",

              loss="categorical_crossentropy",     

              metrics=["accuracy"])
model.summary()
tracker = model.fit(X_train, y_train,

                    batch_size=400,

                    epochs=30,

                    validation_data=(X_validate, y_validate),

                    verbose=0)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])

print('Test accuracy:', score[1])

# around: 0.93
fig, ax = plt.subplots(figsize = (8,6))

ax.plot(tracker.history["loss"], label = "training_loss")

ax.plot(tracker.history["val_loss"], label = "val_loss")

plt.xlabel("epochs")

plt.ylabel("loss function")

ax.legend(loc = 'upper center', shadow = True,)

plt.show()
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(tracker.history["acc"], label = "training_accuracy")

ax.plot(tracker.history["val_acc"], label = "val_accuracy")

plt.xlabel("epochs")

plt.ylabel("accuracy")

ax.legend(loc = 'best', shadow = True,)

plt.show()