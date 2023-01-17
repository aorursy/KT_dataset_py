import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras import optimizers
from keras.optimizers import RMSprop, Adagrad, Adadelta
import matplotlib.pyplot as plt
df = pd.read_csv('../input/digit-recognizer/train.csv')
pixels = df.columns.drop('label')
y_column = 'label'
df.head()
df[pixels] = df[pixels].applymap(lambda x: x / 255)
train, validation = train_test_split(df, test_size=0.2)
x_train = train[pixels].values
y_train = train[y_column].values

x_val = validation[pixels].values
y_val = validation[y_column].values
y_train = y_train.reshape((y_train.shape[0], 1))
y_val = y_val.reshape((y_val.shape[0], 1))
print(x_train.shape, y_train.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_t = lr.predict(x_val)
from sklearn.metrics import accuracy_score
accuracy_score(y_val, y_t)
def plotgraph(history):
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color = 'b', label = "Training loss")
    ax[0].plot(history.history['val_loss'], color = 'r', label = "Validation loss", axes = ax[0])
    legend = ax[0].legend(loc = 'best', shadow = True)
    ax[1].plot(history.history['acc'], color = 'b', label = "Training accurancy")
    ax[1].plot(history.history['val_acc'], color = 'r', label = "Validation accurancy")
    legend = ax[1].legend(loc = 'best', shadow = True)
model1 = Sequential()
model1.add(Dense(units = 64, activation = 'relu', input_dim = len(pixels)))
model1.add(Dense(units = 32, activation = 'relu'))
model1.add(Dropout(0.25))
model1.add(Dense(units = 10, activation = 'softmax'))
optimizer = RMSprop (lr = 0.001) #сколрость обучения
model1.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
#history1 = model1.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 50, batch_size = 32)
#plotgraph(history)
model2 = Sequential()
model2.add(Dense(256, activation = 'relu', input_dim = len(pixels)))
model2.add(Dense(128, activation = 'relu'))
model2.add(Dense(64, activation = 'relu'))
model2.add(Dropout(0.25))
model2.add(Dense(10, activation = 'softmax'))
optimizer2 = Adagrad()
model2.compile(optimizer = optimizer2, loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
#history2 = model2.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 50, batch_size = 32)
#plotgraph(history2)
optimizer3 = Adadelta()
model2.compile(optimizer = optimizer3, loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
#history3 = model2.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 50, batch_size = 32)
#plotgraph(history3)
from PIL import Image
import numpy as np
import sys
import os
import csv

#Useful function
def createFileList(myDir, format='.png'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# load the original image
fileList = createFileList('../input/images-dr/')

for file in fileList:
    print(file)
    img_file = Image.open(file)
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    print(value)
    with open("img_pixels.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)
