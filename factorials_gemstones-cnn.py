# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
#
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import cv2
train_input = []
test_input = []
train_label = []
test_label = []
root_folder = "/kaggle/input";
for folder_name in os.listdir(root_folder) :
    name = root_folder + "/" + folder_name
    for folder in os.listdir(name) :
        path_name = name + "/" + folder
        for label_name in os.listdir(path_name) :
            label_folder = path_name + "/" + label_name
            for file_name in os.listdir(label_folder) :
                image = cv2.imread(label_folder + "/" + file_name)
                image = cv2.resize(image,(256,256))
                
                if folder == "train" :
                    train_input.append(image)
                    train_label.append(label_name)
                elif folder == "test" :
                    test_input.append(image)
                    test_label.append(label_name)
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
train_input = np.array(train_input)
train_input = train_input.astype('float32')
train_input /= 255

test_input = np.array(test_input)
test_input = test_input.astype('float32')
test_input /= 255

enc = LabelEncoder()
p = enc.fit_transform(train_label)
train_label = to_categorical(p)

enc = LabelEncoder()
p = enc.fit_transform(test_label)
test_label = to_categorical(p)






model = Sequential()

model.add(Conv2D(256,(4, 4), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128,(4,4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,(4,4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,(4,4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))


model.add(Dense(87, activation='softmax'))

model.summary()
from keras.optimizers import Adam
model.compile(loss = 'binary_crossentropy',optimizer = Adam(lr = 0.001))

model.fit(train_input,train_label, validation_split = 0.1,epochs=100)
acc = model.evaluate(test_input,test_label)
print(acc)