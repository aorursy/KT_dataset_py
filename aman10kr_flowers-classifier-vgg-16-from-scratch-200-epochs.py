import numpy as np
import pandas as pd
import os
import cv2
import keras
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras import backend as K
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Activation, Flatten, Dense,MaxPooling2D, Dropout
from keras.layers import Conv2D,MaxPool2D, BatchNormalization
from keras.layers import Input, GlobalAveragePooling2D
from keras import models
from keras.models import Model
img_size = 124
train_data = []
val_data = []
test_data = []
directory = "../input/flowers-recognition/flowers/flowers"
for sub_directory in os.listdir(directory):
    count = 0
    inner_directory = os.path.join(directory,sub_directory)
    test_limit = int(0.85*(len(os.listdir(inner_directory))))
    val_limit = int(0.8*test_limit)
    for i in os.listdir(inner_directory):
        try:
            count += 1
            img = cv2.imread(os.path.join(inner_directory,i),1)
            img = cv2.resize(img,(img_size,img_size))
            if count < val_limit:
                train_data.append([img,sub_directory])
            elif val_limit <= count < test_limit:
                val_data.append([img,sub_directory])
            else:
                test_data.append([img,sub_directory])
        except:
            pass
random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)
train_X = []
train_Y = []
for features,label in train_data:
    train_X.append(features)
    train_Y.append(label)
val_X = []
val_Y = []
for features,label in val_data:
    val_X.append(features)
    val_Y.append(label)

test_X = []
test_Y = []
for features,label in test_data:
    test_X.append(features)
    test_Y.append(label)
train_X = np.array(train_X)/255.0
train_X = train_X.reshape(-1,124,124,3)
train_Y = np.array(train_Y)
val_X = np.array(val_X)/255.0
val_X = val_X.reshape(-1,124,124,3)
val_Y = np.array(val_Y)
test_X = np.array(test_X)/255.0
test_X = test_X.reshape(-1,124,124,3)
test_Y = np.array(test_Y)
train_X.shape
w=10
h=10
fig=plt.figure(figsize=(12,12))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    img = train_X[i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(np.squeeze(img))
plt.show()
LE = LabelEncoder()
train_Y = LE.fit_transform(train_Y)
test_Y = LE.fit_transform(test_Y)
val_Y = LE.fit_transform(val_Y)
train_Y = np_utils.to_categorical(train_Y)
test_Y = np_utils.to_categorical(test_Y)
val_Y = np_utils.to_categorical(val_Y)
train_Y.shape
train_Y[0]
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same', input_shape=(124,124,3)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3),padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,    
        rotation_range=20,    
        width_shift_range=0.1,
        height_shift_range=0.1,  
        horizontal_flip=False,  
        vertical_flip=False)
datagen.fit(train_X)
history = model.fit(datagen.flow(train_X,train_Y, batch_size = 64) ,epochs = 200 , validation_data = datagen.flow(val_X, val_Y))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
score = model.evaluate(test_X, test_Y, verbose=0)
print("Loss: " + str(score[0]))
print("Accuracy: " + str(score[1]*100) + "%")