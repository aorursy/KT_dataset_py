# -*- coding: utf-8 -*-



# I don't like warnings, especially user warnings at all!

import warnings

warnings.filterwarnings('ignore')
#Resources: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

#Dataset:https://bengali.ai/
# Import some packages that we require

import pandas as pd

import os

import glob

import umap

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from PIL import Image

from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential, Model

from keras.applications import vgg16

from keras.applications import resnet50

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.optimizers import Adam, RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

from skimage.io import imread, imshow

from skimage.transform import resize

import imgaug as ia

from imgaug import augmenters as iaa

from keras import backend as K

import tensorflow as tf

from collections import defaultdict, Counter

from sklearn.preprocessing import StandardScaler

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.metrics import confusion_matrix

from keras.backend import clear_session

print(os.listdir("../input"))
# For plotting within the notebook

%matplotlib inline



# Graphics in SVG format are more sharp and legible

%config InlineBackend.figure_format = 'svg'



# seaborn color palette 

color = sns.color_palette()



# For REPRODUCIBILITY

seed = 42

np.random.seed(seed)

tf.set_random_seed(seed)
input_path = Path("../input")



# Path to training images and corresponding labels provided as numpy arrays

numtadb_train_path = input_path/"NumthDB_training.npz"



# Path to the test images and corresponding labels

numtadb_test_path = input_path/"NumthDB_test.npz"
train_images = np.load(numtadb_train_path)['data']

train_labels = np.load(numtadb_train_path)['label']



# Load the test data from the corresponding npz files

test_images = np.load(numtadb_test_path)['data']
print(f"Number of training samples: {len(train_images)} where each sample is of size: {train_images.shape[1:]}")

print(f"Number of test samples: {len(test_images)} where each sample is of size: {test_images.shape[1:]}")
# Get the unique labelsa

labels = np.unique(train_labels)

print("No of uniqe lables",labels.shape)

# Get the frequency count for each label

frequency_count = np.bincount(train_labels)



# Visualize 

plt.figure(figsize=(10,5))

sns.barplot(x=labels, y=frequency_count);

plt.title("Distribution of labels in KMNIST training data", fontsize=16)

plt.xlabel("Labels", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.show()
random_samples = []

for i in range(10):

    samples = train_images[np.where(train_labels==i)][:3]

    random_samples.append(samples)



# Converting list into a numpy array

random_samples = np.array(random_samples)



# Visualize the samples

f, ax = plt.subplots(10,3, figsize=(10,20))

for i, j in enumerate(random_samples):

    ax[i, 0].imshow(random_samples[i][0,:,:], cmap='gray')

    ax[i, 1].imshow(random_samples[i][1,:,:], cmap='gray')

    ax[i, 2].imshow(random_samples[i][2,:,:], cmap='gray')

    

    ax[i,0].set_title(str(i))

    ax[i,0].axis('off')

    ax[i,0].set_aspect('equal')

    

    ax[i,1].set_title(str(i))

    ax[i,1].axis('off')

    ax[i,1].set_aspect('equal')

    

    ax[i,2].set_title(str(i))

    ax[i,2].axis('off')

    ax[i,2].set_aspect('equal')

plt.show()
def get_random_samples(nb_indices):

    # Choose indices randomly 

    random_indices = np.random.choice(nb_indices, size=nb_indices, replace=False)



    # Get the data corresponding to these indices

    random_train_images = train_images[random_indices].astype(np.float32)

    random_train_images /=255.

    random_train_images = random_train_images.reshape(nb_indices, 28*28)

    random_train_labels = train_labels[random_indices]

    labels = np.unique(random_train_labels)

    return random_indices, random_train_images, random_train_labels, labels
#Get randomly sampled data

nb_indices = 5000

random_indices, random_train_images, random_train_labels, labels = get_random_samples(nb_indices)



# Get the actual labels from the labels dictionary

labels_name = [x for x in labels]



# Get a t-SNE instance

#Reduced n_components=2

tsne = TSNE(n_components=2, random_state=seed, perplexity=30,n_iter=2000)



# Do the scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(random_train_images)



# Fit tsne to the data

random_train_2D = tsne.fit_transform(random_train_images)

print("Shape of the new manifold",random_train_2D.shape)
plot_num = 500

fig = plt.figure(figsize=(10, 8))

for i, label in zip(labels, labels_name):

    sns.scatterplot(random_train_2D[random_train_labels == i, 0][:plot_num], 

                random_train_2D[random_train_labels == i, 1][:plot_num], 

                label=i, s=18)



plt.title("Visualizating NumtaDb embeddings using tSNE", fontsize=16)

plt.legend()

plt.show()
# Let's try UMAP now.

nb_indices = 50000

random_indices, random_train_images, random_train_labels, labels = get_random_samples(nb_indices)

#Reduced n_components=2

embedding = umap.UMAP(n_components=2, metric='correlation', min_dist=0.8)

random_train_2D = embedding.fit_transform(random_train_images)



fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111) #projection='3d')



for i, label in zip(labels, labels):

    sns.scatterplot(random_train_2D[random_train_labels == i, 0], 

                random_train_2D[random_train_labels == i, 1], 

                label=label, s=15)

plt.title("Visualiza NumtaDb embeddings using UMAP ", fontsize=16)

plt.legend()

plt.show()
print(train_images.shape)

train_images_1d = train_images.reshape((-1,784))

print(train_images_1d.shape)
print(test_images.shape)

test_images_1d = test_images.reshape((-1,28,28,1))

print(test_images_1d.shape)
train_labels_encoded = to_categorical(train_labels,num_classes=10)

print(train_labels_encoded.shape)
X_train, X_val, Y_train, Y_val = train_test_split(train_images_1d, train_labels_encoded, test_size = 0.2, random_state=seed)
#Clear last session and start a fresh session

from keras import backend as K

K.clear_session()

sess = tf.Session()

K.set_session(sess)
model = Sequential()

model.add(Dropout(0.2, input_shape=(784,)))

model.add(Dense(units=512,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=10,activation='softmax'))

# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
%%time

epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 64

# Without data augmentation i obtained an accuracy of 0.98114

history = model.fit(

    X_train, Y_train,

    batch_size = batch_size, 

    epochs = epochs, 

    validation_data = (X_val, Y_val), 

    verbose = 2,

    callbacks=[learning_rate_reduction])
print(train_images.shape)

train_images_3d = train_images.reshape((-1,28,28,1))

print(train_images_3d.shape)
print(test_images.shape)

test_images_3d = test_images.reshape((-1,28,28,1))

print(test_images_3d.shape)
train_labels_encoded = to_categorical(train_labels,num_classes=10)

print(train_labels_encoded.shape)
X_train, X_val, Y_train, Y_val = train_test_split(train_images_3d, train_labels_encoded, test_size = 0.2, random_state=seed)
#Clear last session and start a fresh session

from keras import backend as K

K.clear_session()

sess = tf.Session()

K.set_session(sess)
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
%%time

epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 64

# Without data augmentation i obtained an accuracy of 0.98114

history = model.fit(

    X_train, Y_train,

    batch_size = batch_size, 

    epochs = epochs, 

    validation_data = (X_val, Y_val), 

    verbose = 2,

    callbacks=[learning_rate_reduction])
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
# Display some error results 

# Errors are difference between predicted labels and true labels



errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 3

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-9:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

## Model is getting confused betwwen 1 and 9,which is decent
# predict results

test_images_3d = test_images.reshape((-1,28,28,1))

results = model.predict(test_images_3d)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
#Clear and start a fresh session

from keras import backend as K

K.clear_session()

sess = tf.Session()

K.set_session(sess)
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
epochs = 30

batch_size = 64

# Fit the model

history = model.fit_generator(

    datagen.flow(X_train,Y_train, batch_size=batch_size),

    epochs = epochs, 

    validation_data = (X_val,Y_val),

    verbose = 2,

    steps_per_epoch=X_train.shape[0] // batch_size,

    callbacks=[learning_rate_reduction]

)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
## Save the model

model.save_weights('model_weights.h5')



# Save the model architecture

with open('model_architecture.json', 'w') as f:

    f.write(model.to_json())
#Prepare test data

test_images_3d = test_images.reshape((-1,28,28,1))

# predict results

results = model.predict(test_images_3d)

# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,test_images_3d.shape[0]+1),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_numta_datagen.csv",index=False)