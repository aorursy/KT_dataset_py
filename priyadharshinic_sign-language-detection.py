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
# Imports
import keras
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
# Fetching data
train = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")
train.info()
test.info()
train.head()
# The column 'label' (1 to 24) alone loaded in separate dataframes
train_label = train['label']
test_label = test['label']
# Dropping the label column in training set so as to contain only pixel values for each label
train = train.drop(['label'],axis=1)
train.head()
# Converting 1-D to 3-D array to use CNN model
x_train = train.values
x_train = x_train.reshape(-1,28,28,1)
print(x_train.shape)
# Similarly for test data
x_test = test.drop(['label'],axis=1)
x_test = x_test.values.reshape(-1,28,28,1)
print(x_test.shape)
# Frequency plot of each label
sns.countplot(train_label)
plt.title("Frequency of each label")
lb = LabelBinarizer()
y_train = lb.fit_transform(train_label)
y_test = lb.fit_transform(test_label)
y_train.shape
y_test.shape
# Generating new data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 0,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

x_test=x_test/255
# Building CNN model
model = Sequential()
model.add(Conv2D(128, kernel_size=(5,5), strides=1, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Conv2D(64, kernel_size=(2,2), strides=1, activation='relu', padding='same'))
model.add(MaxPool2D((2,2), 2, padding='same'))
model.add(Conv2D(32, kernel_size=(2,2), strides=1, activation='relu', padding='same'))
model.add(MaxPool2D((2,2), 2, padding='same'))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24, activation='softmax'))
model.summary()
# Compiling the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Training the model
model.fit(train_datagen.flow(x_train,y_train,batch_size=200), epochs = 30, validation_data = (x_test,y_test), shuffle=1)
# Loss and Accuracy
(loss,accuracy) = model.evaluate(x=x_test,y=y_test)
print('Accuracy = {}%'.format(accuracy*100))
print('Loss = {}%'.format(loss*100))