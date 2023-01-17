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
from keras.layers import Flatten,Activation
from keras.layers import Dense, Dropout
from numpy import asarray
from PIL import Image
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D
#labelling data and reading images from the dataset(training dataset)
filenames = (os.listdir("../input/dogs-vs-cats/train/train/"))
y_train = []
x_train = []
j=0
for i in filenames:
    j+=1
    photo = load_img("../input/dogs-vs-cats/train/train/"+i,target_size=(224,224))
    photo = img_to_array(photo)
    x_train.append(photo)
    if 'cat' in i:
        y_train.append(0)
    else:
        y_train.append(1)
    if j>10000:
        break
x_train = asarray(x_train)
y_train = np.array(y_train)
#count plot of ratio of number of cats vs dogs images
import seaborn as sns
sns.countplot(x=pd.DataFrame(y_train)[0], data=pd.DataFrame(y_train))
#labelling data and reading images from the dataset(test dataset)
filenames = (os.listdir("../input/cat-and-dog/test_set/test_set/cats/"))
y_test = []
x_test = []
k=0
for i in filenames:
    if i != '_DS_Store':
        photo = load_img("../input/cat-and-dog/test_set/test_set/cats/"+i,target_size=(224,224))
        photo = img_to_array(photo)
        x_test.append(photo)
        if 'cat' in i:
            y_test.append(0)
        else:
            y_test.append(1)
        if k>2000:
            break
filenames = (os.listdir("../input/cat-and-dog/test_set/test_set/dogs/"))
l=0
for i in filenames:
    if i != '_DS_Store':
        photo = load_img("../input/cat-and-dog/test_set/test_set/dogs/"+i,target_size=(224, 224))
        photo = img_to_array(photo)
        x_test.append(photo)
        if 'cat' in i:
            y_test.append(0)
        else:
            y_test.append(1)
        if l>2000:
            break
x_test = asarray(x_test)
y_test = np.array(y_test)
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape)
print(y_train.shape)
#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, batch_size=64,epochs=20,verbose=1,validation_split=0.20)
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
