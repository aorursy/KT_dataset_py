import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Image



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop, adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.utils import plot_model



%matplotlib inline
train_df = pd.read_csv("../input/fashion-mnist_train.csv")

test_df = pd.read_csv("../input/fashion-mnist_test.csv")

print("training set shape:", train_df.shape)

print("testing set shape:", test_df.shape)
print("Missing Data in training data:", train_df.isnull().values.any())

print("Missing Data in testing data:", test_df.isnull().values.any())
train_df.head()
X_train = train_df.drop(["label"], axis = 1)

Y_train = train_df.label



X_test = test_df.drop(["label"], axis = 1)

Y_test = test_df.label



del train_df, test_df
print("No. of training observations:", len(X_train))

print("No. of testing observations:", len(X_test))

print("No. of distict classes available in training set:", len(set(Y_train)))

print("No. of distict classes available in testing set:", len(set(Y_test)))
sns.countplot(Y_train)
sns.countplot(Y_test)
Y_train = to_categorical(Y_train, 10)

#Y_test = to_categorical(Y_test, 10)
X_train = X_train / 255.0

X_test = X_test / 255.0
X_train.head()
X_test.head()
X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)
X_train.shape
random_seed = 100

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = random_seed)
plt.imshow(X_train[0][:,:,0])
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', 

                 input_shape = (28,28,1), name = "CONV_1"))

model.add(BatchNormalization(name = "BN_1"))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', name = "CONV_2"))

model.add(BatchNormalization(name = "BN_2"))

model.add(MaxPool2D(pool_size=(2,2), name = "MAXPOOL_1"))

model.add(Dropout(0.20, name = "DROP_1"))



model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu', name = "CONV_3"))

model.add(BatchNormalization(name = "BN_3"))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu', name = "CONV_4"))

model.add(BatchNormalization(name = "BN_4"))

model.add(MaxPool2D(pool_size=(2,2), name = "MAXPOOL_2"))

model.add(Dropout(0.20, name = "DROP_2"))



model.add(Flatten(name = "FLAT_1"))

model.add(Dense(128, activation = "relu", name = "FC_1"))

model.add(BatchNormalization(name = "BN_5"))

model.add(Dropout(0.20, name = "DROP_3"))

model.add(Dense(10, activation = "softmax", name = "FC_2"))

model.summary()
plot_model(model, to_file='model.png', show_shapes = True, show_layer_names = True)

Image("model.png")
optimizer = adam(lr = 0.001, epsilon = 1e-08, decay = 0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
callbacks = [

    EarlyStopping(

        monitor = 'val_acc', 

        patience = 10,

        mode = 'max',

        verbose = 1),

    ReduceLROnPlateau(

        monitor = 'val_acc', 

        patience = 3, 

        verbose = 1, 

        factor = 0.5, 

        min_lr = 0.00001)]
epochs = 50

batch_size = 64
datagen = ImageDataGenerator(

        rotation_range = 10, 

        zoom_range = 0.1, 

        width_shift_range = 0.1,  

        height_shift_range = 0.1)  



datagen.fit(X_train)
his = model.fit_generator(datagen.flow(X_train, 

                                 Y_train, 

                                 batch_size = batch_size),

                    epochs = epochs, 

                    validation_data = (X_val,Y_val),

                    verbose = 1, 

                    steps_per_epoch = X_train.shape[0] // batch_size,

                    callbacks = callbacks)
fig, ax = plt.subplots(2,1)

ax[0].plot(his.history['loss'], color = 'b', label = "Training loss")

ax[0].plot(his.history['val_loss'], color = 'r', label = "validation loss" ,axes = ax[0])

legend = ax[0].legend(loc = 'best', shadow = True)



ax[1].plot(his.history['acc'], color = 'b', label = "Training accuracy")

ax[1].plot(his.history['val_acc'], color = 'r',label = "Validation accuracy")

legend = ax[1].legend(loc = 'best', shadow = True)
Y_train = np.argmax(Y_train, axis = 1)

trainResults = model.predict(X_train)

trainResults = np.argmax(trainResults, axis = 1)

trainResults = pd.Series(trainResults, name = "Label")



Y_val = np.argmax(Y_val, axis = 1)

validationResults = model.predict(X_val)

validationResults = np.argmax(validationResults, axis = 1)

validationResults = pd.Series(validationResults, name = "Label")



testResults = model.predict(X_test)

testResults = np.argmax(testResults, axis = 1)

testResults = pd.Series(testResults, name = "Label")



trainAccuracy = (sum(Y_train == trainResults)/len(Y_train)) * 100

valAccuracy = (sum(Y_val == validationResults)/len(Y_val)) * 100

testAccuracy = (sum(Y_test == testResults)/len(Y_test)) * 100



print("training Accuracy:", round(trainAccuracy, 2))

print("validation Accuracy:", round(valAccuracy, 2))

print("testing Accuracy:", round(testAccuracy, 2))