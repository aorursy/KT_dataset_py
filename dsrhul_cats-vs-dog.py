# import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from numpy import asarray
from PIL import Image
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D
#labelling data and reading images from the dataset(training dataset)
filenames = (os.listdir("../input/dogs-vs-cats/train/train/"))
y_train = []
x_train = []
for i in filenames:
    photo = load_img("../input/dogs-vs-cats/train/train/"+i,target_size=(128,128))
    photo = img_to_array(photo)
    x_train.append(photo)
    if 'cat' in i:
        y_train.append(0)
    else:
        y_train.append(1)
x_train = asarray(x_train)
y_train = np.array(y_train)
#count plot of ratio of number of cats vs dogs images
import seaborn as sns
sns.countplot(x=pd.DataFrame(y_train)[0], data=pd.DataFrame(y_train))
#labelling data and reading images from the dataset(test dataset)
filenames = (os.listdir("../input/cat-and-dog/test_set/test_set/cats/"))
y_test = []
x_test = []
for i in filenames:
    if i != '_DS_Store':
        photo = load_img("../input/cat-and-dog/test_set/test_set/cats/"+i,target_size=(128,128))
        photo = img_to_array(photo)
        x_test.append(photo)
        if 'cat' in i:
            y_test.append(0)
        else:
            y_test.append(1)
filenames = (os.listdir("../input/cat-and-dog/test_set/test_set/dogs/"))
for i in filenames:
    if i != '_DS_Store':
        photo = load_img("../input/cat-and-dog/test_set/test_set/dogs/"+i,target_size=(128,128))
        photo = img_to_array(photo)
        x_test.append(photo)
        if 'cat' in i:
            y_test.append(0)
        else:
            y_test.append(1)
x_test = asarray(x_test)
y_test = np.array(y_test)
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape)
print(y_train.shape)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(128,128,3),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model_train = model.fit(x_train, y_train, batch_size=64,epochs=20,verbose=1,validation_split=0.10)
model.fit(x_train, y_train, batch_size=64,epochs=20,verbose=1,validation_split=0.20)
preds = model.predict(x_test, verbose=1)
#model performance evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy score: ",accuracy_score(y_test, np.round_(preds)))
print("Classification report:")
print(classification_report(y_test, np.round_(preds)))
preds = model.predict(x_test, verbose=1)
#model performance evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy score: ",accuracy_score(y_test, np.round_(preds)))
print("Classification report:")
print(classification_report(y_test, np.round_(preds)))
preds = model.predict(x_train, verbose=1)
#model performance evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy score: ",accuracy_score(y_train, np.round_(preds)))
print("Classification report:")
print(classification_report(y_train, np.round_(preds)))
