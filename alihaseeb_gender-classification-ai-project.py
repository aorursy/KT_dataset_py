# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
   # for filename in filenames:
       # print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#the imports
import random
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, Activation, AveragePooling2D
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
%matplotlib inline
men = []
women = []
img_size = 300
MEN_IMGS_PATH = '../input/gender-classification-dataset/Training/male'
WOMEN_IMGS_PATH = '../input/gender-classification-dataset/Training/female'
DIRS = [(0, MEN_IMGS_PATH), (1, WOMEN_IMGS_PATH)]
train_images = []
labels = []
for num, _dir in DIRS:
    _dir = _dir + '/'
    count = 0
    for file in os.listdir(_dir):
        if count >= 1400:
            break
        img = image.load_img(_dir + str(file), target_size=(img_size, img_size))
        img = image.img_to_array(img)
        img = img/255
        train_images.append(img)
        labels.append(num)
        count += 1
train_images[1].shape
plt.imshow(train_images[4])
plt.imshow(train_images[1510])
len(train_images)
X = np.array(train_images)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=101)
print(X_train.shape)
print(X_test.shape)

len(X_train)
len(X_test)
y_train_labels = to_categorical(y_train)
print(y_train_labels.shape)
def build(width, height, depth, classes):
    #initialize the model along with the input shape
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    
    if K.image_data_format() == 'channels_first':
        inputShape = (depth, height, width)
        chanDim = 1
        
    # CONV -> RELU -> MAXPOOL
    model.add(Convolution2D(64, (3,3), padding='same', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))
    
    # (CONV -> RELU)*2 -> AVGPOOL
    model.add(Convolution2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Convolution2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(AveragePooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))
    
    # CONV -> RELU -> MAXPOOL
    model.add(Convolution2D(256, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))
    
    # CONV -> RELU -> AVGPOOL
    model.add(Convolution2D(512, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(AveragePooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))
    
    # DENSE -> RELU
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # DENSE -> RELU
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # sigmoid -> just to check the accuracy with this (softmax would work too)
    model.add(Dense(classes))
    model.add(Activation('sigmoid'))
    
    return model
model = build(img_size, img_size, 3, 2)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train_labels, batch_size=16, epochs=2, validation_split=0.2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
predictions = model.predict_classes(X_test)
print(predictions)
print(confusion_matrix(predictions, y_test))
print(classification_report(predictions, y_test))
random_indices = [random.randint(0, 280) for i in range(9)]
plt.figure(figsize=(10,10))
for i, index in enumerate(random_indices):
    pred = predictions[index]
    pred = 'man' if pred==0 else 'woman'
    actual = 'man' if y_test[index]==0 else 'woman'
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[index], cmap='gray', interpolation='none')
    plt.title(f"Predicted: {pred}, \n Class: {actual}")
    plt.tight_layout()
from sklearn import tree
#DecisionTreeClassifier
#AH
dtc_clf = tree.DecisionTreeClassifier()

X_train = X_train.reshape(2520,300*300*3) 
X_test = X_test.reshape(280,300*300*3) 
# AH.I have reshaped data into 2d for Decision Tree
#AH
dtc_clf = dtc_clf.fit(X_test, y_test)
dtc_prediction = dtc_clf.predict(X_test)
print (dtc_prediction)
#AH
print(confusion_matrix(dtc_prediction, y_test))
print(classification_report(dtc_prediction, y_test))
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=101)
print(X_train.shape)
print(X_test.shape)
random_indices = [random.randint(0, 280) for i in range(9)]
plt.figure(figsize=(10,10))
for i, index in enumerate(random_indices):
    pred = dtc_prediction[index]
    pred = 'man' if pred==0 else 'woman'
    actual = 'man' if y_test[index]==0 else 'woman'
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[index], cmap='gray', interpolation='none')
    plt.title(f"Predicted: {pred}, \n Class: {actual}")
    plt.tight_layout()
    
    
