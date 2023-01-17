# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for preprocessing

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split



# for plotting

import matplotlib.pyplot as plt

import seaborn as sns



# for CNN

import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout



# for image/video recording/loading

import cv2



# for Uniform Manifold Approximation and Projections

from umap import UMAP



# for saving file as json

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pwd
# raw_train_data = pd.read_csv("../input/sign-language-mnist/sign_mnist_train.csv")
# raw_train_data.head()
# labels = raw_train_data['label'].values #saving labels to a numpy ndarray before dropping them

# train_data = raw_train_data.drop(['label'],axis=1) #dropping labels (target value)
# train_data.head()
# train_data = train_data.values # converting to numpy ndarray
# train_data
# label_binarizer = LabelBinarizer()

# labels_encoded = label_binarizer.fit_transform(labels)
# labels_encoded # binarized labels
# plt.imshow(train_data[0].reshape(28,28), cmap='binary') #showing the first row in train_data as an image

# plt.grid(False)
# labels_alpha = []

# def labels_to_alpha(labels):

#     for x in range(labels.size):

#             labels_alpha.append(chr(labels[x]+65))

#     return labels_alpha
# np.unique(labels)
# np.unique(labels_to_alpha(labels))
# num_of_classes = np.unique(labels).size

# num_of_classes
# plt.figure(figsize = (18,8))

# plt.title=("count plot of signs") # TODO this is not showing up in plot

# sns.countplot(x = labels_alpha, order = np.unique(labels_alpha))
# X_train, X_test, Y_train, Y_test = train_test_split(train_data, labels_encoded, test_size = 0.25, random_state = 0)
# X_train_normalized = X_train/255

# X_test_normalized = X_test/255
# fig, (ax1, ax2) = plt.subplots(1, 2)

# ax1.imshow(X_train_normalized[0].reshape(28,28),cmap='binary')

# ax1.set_title("Normalized")

# ax1.grid(False)

# ax2.imshow(X_train[0].reshape(28,28), cmap='binary')

# ax2.set_title("Un-normalized")

# ax2.grid(False)
train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')

test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')
labels = train['label'].values
unique_val = np.array(labels)

np.unique(unique_val)
train.drop('label', axis = 1, inplace = True)
images = train.values

images = np.array([np.reshape(i, (28, 28)) for i in images])

images = np.array([i.flatten() for i in images])
from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

labels = label_binrizer.fit_transform(labels)
labels
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)
batch_size = 128

num_classes = 24

epochs = 50
x_train = x_train / 255

x_test = x_test / 255
x_train_t = np.stack([x_train.reshape(x_train.shape[0],28,28)]*3, axis=3).reshape(x_train.shape[0],28,28,3)

x_test_t = np.stack([x_test.reshape(x_test.shape[0],28,28)]*3, axis=3).reshape(x_test.shape[0],28,28,3)

x_train_t.shape, x_test_t.shape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# Resize the images 48*48 as required by VGG16

from keras.preprocessing.image import img_to_array, array_to_img

x_train_tt = np.asarray([img_to_array(array_to_img(im, scale=True).resize((48,48))) for im in x_train_t])/225

x_test_tt = np.asarray([img_to_array(array_to_img(im, scale=True).resize((48,48))) for im in x_test_t])/225

x_train_tt.shape, x_test_tt.shape
plt.imshow(x_test_tt[0].reshape(48,48,3))
from keras.applications import VGG16

from keras.applications.vgg16 import preprocess_input

modelVGG = Sequential()

#  Create base model of VGG16

modelVGG.add(VGG16(weights='imagenet',

                  include_top=False, pooling = 'avg',  

                  input_shape=(48, 48, 3)

                 ))

# 2nd layer as Dense 

modelVGG.add(Dense(num_classes, activation = 'softmax'))



# Say not to train first layer (ResNet) model as it is already trained

modelVGG.layers[0].trainable = False

modelVGG.summary()
# CNN_model = Sequential()

# CNN_model.add(Conv2D(64,kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))

# CNN_model.add(MaxPooling2D(pool_size=(2,2)))



# CNN_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

# CNN_model.add(MaxPooling2D(pool_size = (2,2)))



# CNN_model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))

# CNN_model.add(MaxPooling2D(pool_size = (2,2)))



# CNN_model.add(Flatten())



# CNN_model.add(Dense(128, activation='relu'))

# CNN_model.add(Dropout(0.20))

# CNN_model.add(Dense(num_of_classes, activation='softmax'))
# X_train_reshaped = X_train_normalized.reshape(X_train_normalized.shape[0], 28, 28, 1)

# X_test_reshaped = X_test_normalized.reshape(X_test_normalized.shape[0],28,28,1)
# CNN_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
modelVGG.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
# batch_size = 128

# epochs = 50
#history = CNN_model.fit(X_train_reshaped, Y_train, validation_data=(X_test_reshaped,Y_test),epochs=epochs,batch_size=batch_size)
historyVGG = modelVGG.fit(x_train_tt, y_train, validation_data = (x_test_tt, y_test), epochs=epochs, batch_size=batch_size)
# CNN_model.save_weights("cnn_weights_50epochs.h5")

# CNN_model.save("cnn_architecture_weights_50epoch")
modelVGG.save_weights("modelvgg_weights_50epochs.h5")

modelVGG.save("modelvgg_architecture_weights_50epoch")
!pwd

!ls
# history.history.keys()
# plt.figure(figsize = (18,8))

# plt.plot(history.history['accuracy'])

# plt.plot(history.history['val_accuracy'])

# plt.xlabel('epoch')

# plt.ylabel('accuracy')

# plt.legend(['train','test'])

# plt.show()
plt.plot(historyVGG.history['accuracy'])

plt.plot(historyVGG.history['val_accuracy'])

plt.title("Accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','test'])

plt.show()
# plt.figure(figsize = (18,8))

# plt.plot(history.history['loss'])

# plt.plot(history.history['val_loss'])

# plt.xlabel('epoch')

# plt.ylabel('loss')

# plt.legend(['train','test'])

# plt.show()
plt.figure(figsize = (18,8))

plt.plot(historyVGG.history['loss'])

plt.plot(historyVGG.history['val_loss'])

plt.title("Loss")

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(['train','test'])

plt.show()
# history.history['accuracy'][-1]*100
historyVGG.history['accuracy'][-1]*100
# history.history['val_accuracy'][-1]*100
historyVGG.history['val_accuracy'][-1]*100
# history.history['loss'][-1]*100
historyVGG.history['loss'][-1]*100
# history.history['val_loss'][-1]*100
historyVGG.history['val_loss'][-1]*100
# TODO save history.history dict which has a list of floats and floats should be converted to str before saving

# with open('history_CNN_model_50epochs.json', 'w') as fp:

#     json.dump(history.history, fp)
test_labels = test['label']

test.drop('label', axis = 1, inplace = True)

test_images = test.values

test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])

#test_images = np.array([i.flatten() for i in test_images])

test_images_t = np.stack([test_images.reshape(test_images.shape[0],28,28)]*3, axis=3).reshape(test_images.shape[0],28,28,3)



# Resize the images 48*48 as required by VGG16

from keras.preprocessing.image import img_to_array, array_to_img

test_images_tt = np.asarray([img_to_array(array_to_img(im, scale=True).resize((48,48))) for im in test_images_t])/225

test_images_tt.shape
plt.imshow(test_images_tt[0].reshape(48,48,3))
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images_tt.reshape(test_images.shape[0], 48, 48, 3)
test_images.shape
# Y_pred = CNN_model.predict(test_data)

# accuracy_score(test_labels_encoded,Y_pred.round())*100
y_pred_vgg = modelVGG.predict(test_images)
accuracy_score(test_labels, y_pred_vgg.round())
# !ls ../input/asl-sign-recognizer/
# CNN_model = keras.models.load_model('../input/asl-sign-recognizer/cnn_architecture_weights_50epoch')
# CNN_model.summary()
# Y_pred.round()[0]
# unique_labels = [x for x in "ABCDEFGHIKLMNOPQRSTUVWXY"]

# unique_labels
# Y_pred_list_alpha = []

# for x in range(Y_pred.round().shape[0]):

#     Y_pred_list_alpha.append(unique_labels[np.argmax(Y_pred.round()[x])])

# Y_pred_list_alpha
# len(Y_pred_list_alpha)
# len(test_labels_encoded)
# test_labels_encoded_alpha = []

# for x in range(test_labels_encoded.shape[0]):

#     test_labels_encoded_alpha.append(unique_labels[np.argmax(test_labels_encoded[x])])

# test_labels_encoded_alpha
# if Y_pred_list_alpha == test_labels_encoded_alpha:

#     print("d")
# cm = confusion_matrix(test_labels_encoded_alpha,Y_pred_list_alpha)
# df = pd.DataFrame(cm ,index = unique_labels, columns = unique_labels)
# plt.figure(figsize = (24,24))

# sns.heatmap(df, annot=True,cmap="YlGnBu")