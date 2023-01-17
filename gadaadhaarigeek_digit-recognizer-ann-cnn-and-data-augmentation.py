# Data processing

import numpy as np 

import pandas as pd 

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator



# Dealing with warnings

import warnings 

warnings.filterwarnings("ignore")



# Plotting the data

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")



# Layers of NN

from keras.models import Sequential

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.core import Dense, Dropout, Activation, Flatten



# Optimizers 

from keras.optimizers import SGD



# Metrics 

from sklearn.metrics import log_loss, confusion_matrix



import itertools

 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load train and test data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train.head()
# Shape of training data 

train.shape



# 42k images, each with 784 features (pixels) of each images
# Shape of training data 

test.shape



# 28k images, each with 784 features (pixels) of each images
# Checking for any null values in training data

train[train.isna().any(1)]
# Checking for any null values in testing data

test[test.isna().any(1)]
# Count of each label 

sns.countplot(train["label"])
# Showing image 

import random

sample_index = random.choice(range(0, 42000))

sample_image = train.iloc[sample_index, 1:].values.reshape(28, 28)

plt.imshow(sample_image)

plt.grid("off")

plt.show()
# Showing with gray map

plt.imshow(sample_image, cmap=plt.get_cmap("gray"))

plt.grid("off")

plt.show()
# Preparing the data

y = train[["label"]]

X = train.drop(labels=["label"], axis=1)
# Standardizing the values

X = X/255.0
# Making class labels as one hot encoding

y_ohe = to_categorical(y)
# Splitting the data 

X_train, X_test, y_train, y_test = train_test_split(X, y_ohe, stratify=y_ohe, test_size=.3)
# Verifying shape of datasets

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Sequential model is linear stackk of layers.



# Since the model needs to know the shape of input it is receiving, the first layer of model will have the 

# input_shape parameter.



# Dense is our normal fully connected layer.

# Dropout layer is applying dropout to a fraction of neurons in particular layer, that means making their weight as 0.



# activation is relu for hidden layer and softmax for output layer (class probabilities).
# Making a basic vanilla neural network to do classification 

model = Sequential()

model.add(Dense(units = 256, input_shape=(784, ), activation="relu"))

model.add(Dropout(rate=.2))



model.add(Dense(units = 256, activation="relu"))

model.add(Dropout(rate=.2))



model.add(Dense(units = 128, activation="relu"))

model.add(Dropout(rate=.2))



model.add(Dense(units = 64, activation="relu"))

model.add(Dense(units = 10, activation="softmax"))
# Compile the model

# Before compiling the model we can actually set some parameters of whichever optimizer we choose 



model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Summary of the model

model.summary()
history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=30, verbose=1, validation_data=(X_test, y_test))
# Plotting the train and validation set accuracy and error



plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='center')



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='center')

plt.show()
# Predicting the output probabilities

y_pred = model.predict(x=X_test, batch_size=32, verbose=1)
# Calculating log loss 

log_loss(y_test, y_pred)
# The log loss is very good
# Converting the probabilitiies as such that highest value will get 1 else 0

y_pred = np.round(y_pred, 2)

b = np.zeros_like(y_pred)

b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
# Now calcualting the log loss

log_loss(y_test, b)

# It has increased now
# CNN stands for convolutional NN majorly due to the convolution operator we have. It works with 2D input array

# (actually 3D as channel value is also expected). 



# It has majorly two parts convolution and pooling (normally max-pooling, but we have other poolings also).

# In convolution layer, we convolve inout image with something called kernels, which help us identify features of

# images like edges, corners, round shapes etc. The output of this is called feature map.



# Convolution happens through strides, we can have values for strides and we also have padding as kernels will have 

# situation where they will not have data fitting the image properly. 



# We have two types of padding, VALID and SAME. 



# VALID padding means no padding, so in that case there will be loss of information. Valid actually means that it will

# only take valid data for the operatioon and will not add any padding.



# SAME padding means zero padding. Here in this case, we actually add 0 padding to match our kernel size with our data

# we will not lose any information in this case.



# This applies to conv layers and pool layers both.



# In a typical CNN, we can have multiple conv layers followed by pool layers to get the feature maps. In hidden layers

# it is typical to have more kernels to get the complex feature maps of the data.



# CNN lacks one thing which Hinton tried to solve in CapsuleNet and that is relative location of elements of images. 

# CNN is not good with identifying relative location of image components



# Let's talk about channels before we moeve ahead. Channels are basically a dimension which lets CNN identify colorful

# images. For a colored RGB image, there are 3 channels. Any RGB image can be divided into 3 different images of 

# different channels and then they are stacked upon each other. We can also see channels as some aspect of information

# in 3rd dimension. For RGB we have 3 channels and for gray map we have 1 channel.
# Reshaping the data for CNN

X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)
# We will implement classic LeNet from Lecun's paper.

# It has two layers having conv, followed by pooling.



model = Sequential()



# First Conv and Pooling layer

model.add(Conv2D(filters=20, kernel_size=(5, 5), padding="same", input_shape=(28, 28, 1)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))



# Second conv and pooling layer

model.add(Conv2D(filters=50, kernel_size=(5, 5), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))



# First fully connected layer

model.add(Flatten())

model.add(Dense(units=500))

model.add(Activation("relu"))



# Second fully connected layer

model.add(Dense(units=10))

model.add(Activation("softmax"))
from keras.optimizers import SGD

opt = SGD(lr=.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()
history = model.fit(x=X_train, y=y_train, batch_size=128, verbose=1, epochs=30, validation_data=(X_test, y_test))
plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='center')



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='center')

plt.show()
y_pred = model.predict(x=X_test, batch_size=32, verbose=1)



# Calculating log loss 

log_loss(y_test, y_pred)
from keras.preprocessing.image import ImageDataGenerator

# Imagedatagenerator DOES NOT USE ORIGINAL IMAGES, instead it uses all the augmented images which replace original

# images



augdata = ImageDataGenerator(

    rotation_range=30, # rotating images

    zoom_range=0.2, # zoom range

    width_shift_range=0.3, # width shift range 

    height_shift_range=0.3, # height shift

    shear_range=0.20,

    horizontal_flip=True,

    vertical_flip = True, 

    fill_mode="nearest"

)
# steps_per_epoch means number of batch iterations before a training epoch can be marked complete. 

# Normally it should be data_size/batch_size. But when we have huge amount of augmented data, we might want to have

# less number of iterations as it would be time consuming and data is anyway augmented randomly so it should not be 

# a problem of information loss.
# We will create a new model for that
history = model.fit_generator(generator=augdata.flow(x=X_train, y=y_train, batch_size=32), 

                              validation_data=(X_test, y_test), steps_per_epoch=np.ceil(X_train.shape[0]/32), 

                              epochs=30)
plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='best', shadow=True)

# plt.show()



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='best', shadow=True)

plt.show()
y_pred = model.predict(x=X_test, batch_size=32, verbose=1)



# Calculating log loss 

from sklearn.metrics import log_loss

log_loss(y_test, y_pred)
model = Sequential()



# First Conv and Pooling layer

model.add(Conv2D(filters=30, kernel_size=(2, 2), padding="same", input_shape=(28, 28, 1), activation="relu"))

model.add(Conv2D(filters=30, kernel_size=(2, 2), padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))



# Second conv and pooling layer

model.add(Conv2D(filters=50, kernel_size=(4, 4), padding="same", activation="relu"))

model.add(Conv2D(filters=50, kernel_size=(4, 4), padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))



# First fully connected layer

model.add(Flatten())

model.add(Dense(units=500))

model.add(Activation("relu"))



# Second fully connected layer

model.add(Dense(units=10))

model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit_generator(generator=augdata.flow(x=X_train, y=y_train, batch_size=32), 

                              validation_data=(X_test, y_test), steps_per_epoch=np.ceil(X_train.shape[0]/32), 

                              epochs=30)
plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='best', shadow=True)

# plt.show()



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='best', shadow=True)

plt.show()
# We can see that our model is able to generalize well, though more accuracy can be achieved if we train for higher

# epoch
y_pred = model.predict(x=X_test, batch_size=32, verbose=1)



# Calculating log loss 

from sklearn.metrics import log_loss

log_loss(y_test, y_pred)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(6, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    plt.grid("off")

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

y_pred = model.predict(X_test)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_test, axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10))
# We can see that 2 is mistaken as 5 a lot and vice versa. Similar is the case with 6 and 9. So augmenting the data 

# in this case, is it good or we should not do things like horizontal or vertical flip etc.
sns.set(style="white")
errors = (y_pred_classes - y_true != 0)

y_pred_classes_errors = y_pred_classes[errors]

y_pred_errors = y_pred[errors]

y_true_errors = y_true[errors]

X_val_errors = X_test[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)), cmap=plt.get_cmap("gray"))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            

            n += 1



# Probabilities of the wrong predicted numbers

y_pred_errors_prob = np.max(y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, y_pred_classes_errors,y_true_errors)