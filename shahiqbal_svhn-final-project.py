# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.constraints import max_norm
from keras.layers import Dense, Dropout, LSTM
from keras.layers import Activation, Flatten, Input, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
# Any results you write to the current directory are saved as output.
def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']

path = '../input/svhntrain/train_32x32.mat'
pathTest = '../input/svhntest/test_32x32.mat'
X_train, y_train = load_data(path)
X_test, y_test = load_data(pathTest)

# Transposing the the train and test data
# by converting it from  
# (width, height, channels, size) -> (size, width, height, channels)

X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]

print("Training Set", X_train.shape)
print('')

print("Test Set", X_test.shape)
print('')

# Calculate the total number of images
num_images = X_train.shape[0]
num_images2 = X_test.shape[0]

print("Total Number of Train Images", num_images)
print("Total Number of Test Images", num_images2)

#Label from 10 to 0
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0
print(np.unique(y_train))

#Validation Set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.13, random_state=7)

train_mean = np.mean(X_train, axis=0) #Shape (32,3)
train_std = np.std(X_train, axis=0) #Shape (32,3)
X_train_norm = (X_train - train_mean) / train_std
X_train_255 = np.divide(X_train, 255)
X_train_255 = np.array(X_train_255)
print("X_train_norm Shape: ", X_train_norm.shape)
print("X_train_255 Shape: ", X_train_255.shape)



X_train_norm_array = np.array(X_train_norm)
y_train_array = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_val_array = np.array(X_val)
y_val_array = np.array(y_val)

print("Y Train Shape: ", y_train_array.shape)

X_test_255 = np.divide(X_test, 255)

y_train_cnn = to_categorical(y_train_array, 10)
y_val_cnn = to_categorical(y_val_array, 10)
y_test_cnn = to_categorical(y_test, 10)
print("Y Train CNN Shape: ", y_train_cnn.shape)
print("Y Val CNN Shape: ", y_val_cnn.shape)
print("Y Test CNN Shape: ", y_test_cnn.shape)
def plot_images(img, labels, nrows, ncols):
    """ Plot nrows x ncols images
    """
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])
plot_images(X_train, y_train, 1, 5)
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

fig.suptitle('Class Distribution', fontsize=14, fontweight='bold', y=1.05)

ax1.hist(y_train, bins=10)
ax1.set_title("Training set")
ax1.set_xlim(1, 10)

ax2.hist(y_test, color='g', bins=10)
ax2.set_title("Test set")

fig.tight_layout()
def cnn_model_convAndMaxPool(): 
    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides=1, activation='relu',
                            input_shape=(32, 32, 3), kernel_constraint=max_norm(3)))
#     model.add(Dropout(0.90))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
#     model.add(Dropout(0.75))
#     model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=1, kernel_constraint=max_norm(3)))
#     model.add(Dropout(0.75))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
#     model.add(Dropout(0.50))
#     model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=1, kernel_constraint=max_norm(3)))
#     model.add(Dropout(0.50))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
#     model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(10))
#     model.add(Activation('relu'))
#     #model.add(Dropout(0.50))
#     model.add(Dense(10))
    model.add(Activation('softmax'))
#     model.add(Dense(10))
#     model.add(Activation('softmax'))
    
    return model;


    
X_test_255.shape, X_train_255.shape
#Accuracy = 0.9319683466502766
def cnn_model_convMaxPoolDropoutEveryLayer():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, input_shape=(32,32,3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.50))

    model.add(Conv2D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.30))

    model.add(Conv2D(128, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.30))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.10))

    model.add(Dense(10, activation='softmax'))
    return model
batch_size = 128
nb_classes = 10
nb_epoch = 20
# X_train_array = np.array(X_train)
model = cnn_model_convMaxPoolDropoutEveryLayer()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# model.fit(X_train_norm_array, y_train_cnn, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val_array, y_val_cnn))
model.fit(X_train_255, y_train_cnn, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val_array, y_val_cnn))
# score = model.evaluate(X_test, y_test_cnn, verbose=0)
score = model.evaluate(X_test_255, y_test_cnn, verbose=0)
print('loss:', score[0])
print('Test accuracy:', score[1])