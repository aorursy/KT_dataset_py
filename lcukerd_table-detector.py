import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from random import shuffle

# imports needed for CNN
import csv
import cv2
import os, glob
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import time
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt

from subprocess import check_output
def parseXML(xmlfile,image_dict,images,labels):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    for table in root.findall('table'):
        page = table.find('region').get('page')
        image_name = xmlfile.replace('reg.xml','') + page + '.ppm'
        if image_name in image_dict:
            images.append(image_dict[image_name])
            image_dict.pop(image_name,None)
            labels.append(1)
def load_data(data_dir):
    images = []
    labels = []
    image_dict = {os.path.join(data_dir, f):cv2.resize(cv2.imread(os.path.join(data_dir, f)), (310, 438)) for f in os.listdir(data_dir) if f.endswith(".ppm") }
    xml_names = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("reg.xml")]
    for fi in xml_names:
        parseXML(fi,image_dict,images,labels)
    for img in image_dict.values():
        images.append(img)
        labels.append(0)
    return images, labels
images, labels = load_data("../input/data/Data")
images = np.array(images)
labels = np.array(labels)
print (len(images))
print (len(labels))
fig=plt.figure(figsize=(20, 20))
for i in range(95,105):
    fig.add_subplot(5, 2, i-94)
    plt.imshow(images[i])
    print (labels[i])
plt.show()
#Data Shuffling
ind_list = [i for i in range(len(images))]
shuffle(ind_list)
images_S  = images[ind_list, :,:,:]
labels_S = labels[ind_list]
print (len(images_S))
fig=plt.figure(figsize=(20, 20))
for i in range(1,11):
    fig.add_subplot(5, 2, i)
    plt.imshow(images_S[i])
    print (labels_S[i])
plt.show()
#hot encoding
labels_S = np_utils.to_categorical(labels_S)
labels_S = labels_S.astype(int)
print (labels_S[90:110])
#Normalization
images_S = np.array(images_S).astype('int')
images_S = images_S / 255
X_train, X_test, y_train, y_test = train_test_split(
            images_S, labels_S, test_size=0.2, random_state=0)
def createCNNModel(num_classes):
    model = Sequential()
    model.add(Convolution2D(20, 4, 4, input_shape=(438, 310, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(15, 4, 4, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu',W_constraint=maxnorm(3)))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    lrate = 0.000001
    epochs = 30
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model
model = createCNNModel(2)
print (len(X_train),len(X_test),len(y_train),len(y_test))
seed = 2
np.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=30)
## I was not able to make a complete or trained model.