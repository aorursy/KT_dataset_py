import pandas as pd

import numpy as np

import os

from glob import glob

import itertools

import fnmatch

import random

import matplotlib.pylab as plt

import seaborn as sns

import cv2

import sklearn

from sklearn import model_selection

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV

from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import keras

from keras import backend as K

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.models import Sequential, model_from_json

from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D,GlobalAveragePooling2D

%matplotlib inline

import tensorflow as tf

import tensorflow_datasets as tfdb

import keras

from keras import callbacks

from keras import optimizers

from keras.engine import Model

from keras.layers import Dropout, Flatten, Dense

from tensorflow.keras.layers import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils

import scipy
base_model = keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(150,150, 3))


from keras.preprocessing.image import ImageDataGenerator



base_dir = '../input/caltech101/Caltech101'



train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'eval')

test_dir = os.path.join(base_dir, 'test')



datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20



def extract_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, 4, 4, 512))

    labels = np.zeros(shape=(sample_count, 101))

    generator = datagen.flow_from_directory(

        directory,

        target_size=(150, 150),

        batch_size=batch_size,

        class_mode='categorical')

    i = 0

    for inputs_batch, labels_batch in generator:

        features_batch = base_model.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1

        if i * batch_size >= sample_count:

            # Note that since generators yield data indefinitely in a loop,

            # we must `break` after every image has been seen once.

            break

    return features, labels



train_features, train_labels = extract_features(train_dir, 6162)

validation_features, validation_labels = extract_features(validation_dir, 820)

test_features, test_labels = extract_features(test_dir, 1695)
train_features = np.reshape(train_features, (6162, 4 * 4 * 512))

validation_features = np.reshape(validation_features, (820, 4 * 4 * 512))

test_features = np.reshape(test_features, (1695, 4 * 4 * 512))
from keras import models

from keras import layers

from keras import optimizers



model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_dim=4 * 4 * 512))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(101, activation='softmax'))
model.summary()
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta

model.compile(loss='categorical_crossentropy',

              optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),

              metrics=['acc'])
history = model.fit(train_features, train_labels,

                    epochs=300,

                    batch_size=20,

                    validation_data=(validation_features, validation_labels))
test_loss, test_acc = model.evaluate(test_features,test_labels)

print("Test Loss: ", test_loss*100)

print("Test Accuracy: ", test_acc*100)
plt.plot(history.history['acc'], 'blue')

plt.plot(history.history['val_acc'], 'orange')

plt.title("Model Accuracyt")

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validate'], loc='upper left')

plt.savefig("Accuracy.png")



print("VGG -Validation Loss: ", history.history['val_loss'][-1]*100)

print("VGG - Validation Accuracy: ", history.history['val_acc'][-1]*100)

print("\n")

print("VGG - Training Loss: ", history.history['loss'][-1]*100)

print("VGG - Training Accuracy: ", history.history['acc'][-1]*100)

print("\n")
plt.plot(history.history['loss'], 'blue')

plt.plot(history.history['val_loss'], 'orange')

plt.title("Model Loss for VGGNet")

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validate'], loc='upper left')

plt.savefig("Loss.png")
from tensorflow.keras import Model

model.save('model_caltech101_vgg19.h5')
target_names = []

for i in range(101):

    a = 'Object '

    b = str(i)

    c = a+b

    c = [i]

    target_names.append((a+b))



def reports(X_test,y_test):

    Y_pred = model.predict(X_test)

    y_pred = np.argmax(Y_pred, axis=1)



    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)

    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

    score = model.evaluate(X_test, y_test, batch_size=32)

    Test_Loss = score[0]*100

    Test_accuracy = score[1]*100

    kc=cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)

    #mse=mean_squared_error(y_test, y_pred)

    #mae=mean_absolute_error(y_test, y_pred)

    #precision=precision_score(y_test, y_pred, average='weighted')

    #print(classification_report(y_test, y_pred, target_names=target_names))





    

    return classification, confusion, Test_Loss, Test_accuracy ,kc#,mse,mae
from sklearn.metrics import classification_report, confusion_matrix,cohen_kappa_score

from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score

# calculate result, loss, accuray and confusion matrix

classification, confusion, Test_loss, Test_accuracy,kc = reports(test_features,test_labels)

classification = str(classification)

confusion_str = str(confusion)
print("confusion matrix: ")

print('{}'.format(confusion_str))

print("KAppa Coeefecient=",kc)

print('Test loss {} (%)'.format(Test_loss))

print('Test accuracy {} (%)'.format(Test_accuracy))

#print("Mean Squared error=",mse)

#print("Mean absolute error=",mae)

print(classification)
import matplotlib.pyplot as plt

%matplotlib inline

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.get_cmap("Blues")):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    Normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if normalize:

        cm = Normalized

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    #print(cm)



    plt.imshow(Normalized, interpolation='nearest', cmap=cmap)

    plt.colorbar()

    plt.title(title)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)



    fmt = '.1f' if normalize else 'd'

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        thresh = cm[i].max() / 2.

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')





plt.figure(figsize=(100,100))

plot_confusion_matrix(confusion, classes=target_names, normalize=False, 

                      title='Confusion matrix, without normalization')

plt.savefig("confusion_matrix_without_normalization.png")

plt.show()

plt.figure(figsize=(100,100))

plot_confusion_matrix(confusion, classes=target_names, normalize=True, 

                      title='Normalized confusion matrix')

plt.savefig("confusion_matrix_with_normalization.png")

plt.show()