# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_csv = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_csv = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(train_csv.shape)
print(test_csv.shape)
train_csv.head(10)
target_train = train_csv['label']
training_csv = train_csv.iloc[:, 1 : ]
training_csv
#training_csv.append(test_csv)
#train_test_csv = training_csv

#features = train_test_csv.shape[1]
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D, BatchNormalization, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt
features_num = training_csv.shape[1]
number_train = training_csv.shape[0]
number_test = test_csv.shape[0]
print(features_num, number_train)
Y_train = to_categorical(target_train, num_classes=10)
x_train = training_csv.values.reshape(number_train, 28, 28, 1)
X_test = test_csv.values.reshape(number_test, 28, 28, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size = 3, activation = 'relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size = 5, activation = 'relu'))
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, Y_train, batch_size = 100, epochs = 40, validation_split=0.3)
def show_curve(history):
    
    fig = plt.figure(figsize=(20,6))
    
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    


show_curve(history)

result = model.predict(X_test)
result = np.argmax(result, axis=1)
result
results = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results], axis = 1)
submission.to_csv("submission.csv", index=False)
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
import seaborn as sns
Y_train = train['label']
X_train = train.drop('label', axis=1)
sns.countplot(Y_train)
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
Y_train = to_categorical(Y_train, num_classes=10)
plt.imshow(X_train[5][:, :, 0], cmap='gray')
num_nets = 10
models = [0] * num_nets
for i in range(num_nets):
    models[i] = Sequential();
    models[i].add(Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same' , input_shape = (28, 28, 1)))
    models[i].add(Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same'))
    models[i].add(MaxPool2D(pool_size = 2))
    models[i].add(BatchNormalization())
    models[i].add(keras.layers.Activation('relu'))
    models[i].add(Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same' , input_shape = (28, 28, 1)))
    models[i].add(Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same'))
    models[i].add(MaxPool2D(pool_size = 2))
    models[i].add(BatchNormalization())
    models[i].add(Dropout(0.4))
    models[i].add(keras.layers.Activation('relu'))
    models[i].add(Conv2D(64, kernel_size = 5, activation = 'relu', padding = 'same' , input_shape = (28, 28, 1)))
    models[i].add(Conv2D(64, kernel_size = 5, activation = 'relu', padding = 'same'))
    models[i].add(MaxPool2D(pool_size = 2))
    models[i].add(BatchNormalization())
    models[i].add(Conv2D(128, kernel_size = 5, activation = 'relu', padding = 'same' , input_shape = (28, 28, 1)))
    models[i].add(BatchNormalization())

    models[i].add(Flatten())
    
    models[i].add(Dense(512, activation = 'relu'))
    models[i].add(Dropout(0.5))
    models[i].add(Dense(10, activation = 'softmax'))
    
    models[i].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
from keras.preprocessing.image import ImageDataGenerator  
data_generator = ImageDataGenerator(featurewise_center=False,  
                                             samplewise_center=False, 
                                             featurewise_std_normalization=False, 
                                             samplewise_std_normalization=False, 
                                             zca_whitening=False, 
                                             zca_epsilon=1e-06, 
                                             rotation_range=10, 
                                             width_shift_range=0.1, 
                                             height_shift_range=0.1, 
                                             brightness_range=None, 
                                             shear_range=0.1, 
                                             zoom_range=0.1, 
                                             channel_shift_range=0.0, 
                                             fill_mode='nearest', 
                                             cval=0.0, 
                                             horizontal_flip=False, 
                                             vertical_flip=False, 
                                             rescale=None, 
                                             preprocessing_function=None, 
                                             data_format=None, 
                                             validation_split=0.3, 
                                             dtype=None)
data_generator.fit(X_train)
from sklearn.model_selection import train_test_split



history = [0] * num_nets
for i in range(num_nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.3)
    history[i] = models[i].fit_generator(data_generator.flow(X_train, Y_train, batch_size = 128), 
                                         steps_per_epoch=len(x_train) / 128, 
                                         validation_data = (X_val2,Y_val2),
                                         epochs = 30 )
def show_curve_multi(history):
    fig = plt.figure(figsize=(28,10))
    for i in range(num_nets):
        plt.subplot(2, 5, i+1)
        plt.plot(history[i].history['accuracy'])
        plt.plot(history[i].history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
show_curve_multi(history)
result = np.zeros((28000, 10))
for i in range(num_nets):
    result += models[i].predict(test)
result = result / 10
result = np.argmax(result, axis = 1)
results = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results], axis = 1)
submission.to_csv("submission_version2.csv", index=False)
