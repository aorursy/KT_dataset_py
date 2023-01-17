import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn import decomposition

from sklearn.model_selection import train_test_split

import time

import os

# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')
X=np.load('../input/Sign-language-digits-dataset/X.npy')

Y=np.load('../input/Sign-language-digits-dataset/Y.npy')
plt.figure(figsize=(10,10))

for i in range(0, 8):

    plt.subplot(440 + 1 + i)

    plt.tight_layout()

    plt.imshow(X[i*100], cmap=plt.get_cmap('gray'))

    plt.axis('off')

    plt.title(Y[i*100].argmax())    

plt.show()
X.shape
Y.shape
num_class = np.unique(Y.argmax(axis=1), return_counts=True) 

plt.title("Number of class")

plt.xticks(num_class[0])

plt.bar(num_class[0], num_class[1],color = (0.2, 0.4, 0.6, 0.6) )

plt.show()
X_flat = np.array(X).reshape((-1, 64*64))

X_train, X_test, y_train, y_test = train_test_split(X_flat, Y, test_size=0.3, random_state=42)
X_flat.shape
from sklearn.decomposition import PCA

pca_dims = PCA()

pca_dims.fit(X_flat)

cumsum = np.cumsum(pca_dims.explained_variance_ratio_)

d = np.argmax(cumsum >= 0.95) + 1

d
pca = PCA(n_components=d)

X_reduced = pca.fit_transform(X_flat)

X_recovered = pca.inverse_transform(X_reduced)
print("reduced shape: " + str(X_reduced.shape))

print("recovered shape: " + str(X_recovered.shape))
f = plt.figure()

f.add_subplot(1,2, 1)

plt.title("original")

plt.imshow(X_train[0].reshape((64,64)))

f.add_subplot(1,2, 2)



plt.title("PCA compressed")

plt.imshow(X_recovered[0].reshape((64,64)))

plt.show(block=True)

X_recovered
X = X.reshape(-1, 64, 64, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=20)
X_train.shape
y_train.shape
X_train = X_train /255

X_test = X_test/255
#X_test
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten,Convolution2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

import keras

from keras import layers,models

from keras.layers import BatchNormalization



from keras import backend as K

from keras.models import Sequential

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dense,Dropout

from keras.utils import np_utils

from keras.layers.convolutional import Conv2D

from keras.optimizers import SGD, RMSprop, Adam
classifier = Sequential()



# Step 1 - Convolution

classifier.add(Conv2D(68, (5, 5), input_shape = (64, 64, 1), activation = 'relu'))



# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

classifier.add(Conv2D(68, (5, 5), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Conv2D(68, (5, 5), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 256, activation = 'relu'))

classifier.add(BatchNormalization())

classifier.add(Dense(units = 10, activation = 'softmax'))



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator



# Generate Images

train_datagen = ImageDataGenerator(

                                   shear_range = 0.08,

                                   zoom_range = 0.08,

                                   horizontal_flip = False,

                                   width_shift_range= 0.02,

                                   height_shift_range= 0.02)

test_datagen = ImageDataGenerator(rescale = 1./255)



# fit parameters from data

training_set = train_datagen.flow(X_train, y_train, batch_size=64)

test_set = test_datagen.flow(X_test, y_test, batch_size=64)



history = classifier.fit_generator(training_set,

                         steps_per_epoch = 60,

                         epochs =20,

                         validation_data = test_set,

                         validation_steps = 500)
scores = classifier.evaluate(X_test, y_test, verbose=0)

print("{}: {:.2f}%".format(classifier.metrics_names[1], scores[1]*100))
import seaborn as sns

from sklearn.metrics import confusion_matrix



# 

# Predict the values from the validation dataset

Y_pred = classifier.predict(X_test)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_test,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="BuPu",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.plot(history.history['loss'], color='r', label="training loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# Plot the accuracy curve for validation 

plt.plot(history.history['val_acc'], color='g', label="validation accuracy")

plt.title("Validation Accuracy")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()