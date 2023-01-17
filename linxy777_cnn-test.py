# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from keras.layers import Conv2D
import keras
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
# Any results you write to the current directory are saved as output.
def read_test_data(filename):
    data = pd.read_csv(filename)
    data = np.array(data)/255.0
    return data.reshape(-1,28,28,1)
    
def read_train_data(filename):
    data = pd.read_csv(filename)
    label = data['label']
    print(label)
    data.drop(columns = ["label"],inplace = True)
    data = np.array(data)/255.0
    return data.reshape(-1,28,28,1),np.array(label)
train_data,train_label = read_train_data("../input/train.csv")
x_train, x_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
test_data = read_test_data("../input/test.csv")
def buildModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Dropout(0.5))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='nadam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
    
datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)
model = buildModel()
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
early_stopping = EarlyStopping(monitor='val_acc', patience=30, verbose=2)
filepath = "best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='auto')
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                           steps_per_epoch=500,
                           epochs=300, #Increase this when not on Kaggle kernel
                           verbose=1,  #1 for ETA, 0 for silent
                           validation_data=(x_val, y_val), #For speed
                           callbacks=[annealer,early_stopping,checkpoint])
from keras.models import load_model
model = load_model("best.hdf5")
final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
import csv
y_hat = model.predict(test_data)
y_pred = np.argmax(y_hat,axis=1)
with open("predict.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ImageId", "Label"])
    for i in range(len(y_pred)):
        writer.writerow([i+1, int(y_pred[i])])