# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import keras

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# Reading train and test CSV data 
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# checking shape of the data
train.shape
# Viewing
train.sample(5, random_state=0)
# Understanding total count of labels and their respective counts
Y_train = train['label']

# Droping 'label' column from train
X_train = train.drop(labels=["label"], axis=1) # {0 or ‘index’, 1 or ‘columns’}

# Deleting train DataFrame
del train

sns.countplot(Y_train)
Y_train.value_counts()
# Check the train data
print(np.isnan(X_train).sum().sum())
X_train.isnull().any().describe()
# Check the train data
test.isnull().any().describe()
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
fig1, ax1 = plt.subplots(1,10, figsize=(10,10))
for i in range(10):
    # reshaping the images into 28*28 shape
    ax1[i].imshow(X_train[i].reshape((28,28)))
    ax1[i].axis('off')
    ax1[i].set_title(Y_train[i]) 
Y_train = to_categorical(Y_train, num_classes=10)
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=555)
# Printing shape of train and validation data
print("X_train Shape: ",X_train.shape)
print("X_val Shape: ",X_val.shape)
print("Y_train Shape: ",Y_train.shape)
print("Y_val Shape: ",Y_val.shape)
# Checking random image
g = plt.imshow(X_train[35][:,:,0])
# The sequential API allows you to create models layer-by-layer for most problems. 
# It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.

model = Sequential() 
# The first convolutional (Conv2D) layer is set as learnable filters.
# set 32 filter and transform a part of the image using kernel filter.
# The kernel filter matrix is applied on the whole image

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))

# The CNN can isolate features that are useful everywhere from these transformed
# images(feature maps)

# The Second important layer in CNN is the pooling(MaxPool2D) layer.
# This layer simply acts as downsampling filter. 
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2))) # we have to choose the area size-
# -pooled each time(pool_size) more the pooling dimension is high, more the
# downsampling is important

# Dropout is a regularization technique for neural network model.
# A simple way to prevent neural network from overfitting
model.add(Dropout(0.25))

# Combining convolutional and pooling layers, CNN are able to combine local
# features and learn more global features of the image
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
# 'relu' is the rectifier(activation function = max(0,x)) is used to add 
# non-linearity to the network
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

# The Flatten layer is used to convert the final feature maps into 
# a one single 1D Vector
model.add(Flatten())

# Two Fully-connected(Dense =  10, activation = softmax) layers
# the net outputs distribution of probability of each class
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# In summary() we can understand the architechture we build and 
# we can understand about the totol number of parameters 
model.summary()

# total_params = (filter_height * filter_width * input_image_channels + 1) * number_of_filters
# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
epochs = 40 
batch_size = 86
# With data augmentation to prevent overfitting

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
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# Look at confusion matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
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
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 