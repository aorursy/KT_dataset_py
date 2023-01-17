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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop


train = pd.read_csv(r"/kaggle/input/digit-recognizer/train.csv",dtype = np.float32)
final_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv", dtype=np.float32)
sample_sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
train.label.head()
train.info()

targets_np = train.label.values
features_np = train.loc[:, train.columns != 'label'].values 


examples_visualize = features_np[1:100].reshape(-1,28,28)
targets_visualize = targets_np[1:100]
rows, columns = 8, 8
fig = plt.figure(figsize=(rows*2,columns*2))
for i in range(rows*columns):
    plt.subplot(rows, columns,i+1)
    plt.tight_layout()
    plt.imshow(examples_visualize[i], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(targets_visualize[i]))
plt.show()

## Statictis classes id in training data 
dict_ids = {}
for idx, value in enumerate(features_np ):
    id = targets_np[idx]    
    if id not in dict_ids:
        dict_ids[id] = 1
    else:
        dict_ids[id] += 1
clasess = list(dict_ids)
objects = (0,1,2,3,4,5,6,7,8,9)
y_pos = np.arange(len(objects))
performance = []
for i in clasess:
    performance.append(dict_ids[i])
fig = plt.figure(figsize=(10,10))

plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Classes id  usage')

plt.show()
targets_np = train.label.values
features_np = train.loc[:, train.columns != 'label'].values / 255.0
features_train, features_val, target_train, target_val = train_test_split(features_np, targets_np, test_size=0.1, random_state=42)

# num_classes = 10
# batch_size = 128
# epochs = 40
# img_rows, img_cols = 28, 28
# x_train = features_train.reshape(features_train.shape[0], img_rows, img_cols, 1)
# x_val = features_val.reshape(features_val.shape[0], img_rows, img_cols, 1)
# input_shape = (img_rows, img_cols, 1)
# y_train = keras.utils.to_categorical(target_train, num_classes)
# y_val = keras.utils.to_categorical(target_val, num_classes)

num_classes = 10
batch_size = 128
epochs = 20
img_rows, img_cols = 28, 28
x_train = features_train.reshape(features_train.shape[0], 784)
x_val = features_val.reshape(features_val.shape[0], 784)
# input_shape = (img_rows, img_cols, 1)
y_train = keras.utils.to_categorical(target_train, num_classes)
y_val = keras.utils.to_categorical(target_val, num_classes)
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(num_classes, activation='softmax'))
# print(model.summary())
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_val, y_val))
# score = model.evaluate(x_val, y_val, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val))
score = model.evaluate(x_val, y_val, verbose=0)
print('Val loss:', score[0])
print('Val accuracy:', score[1])
final_test
final_test_np = final_test.values/255
rows, columns = 8, 8
fig = plt.figure(figsize=(rows*2,columns*2))
for i in range(rows*columns):
    predictions = model.predict_classes(final_test_np[i].reshape(1,784))
    plt.subplot(rows, columns,i+1)
    plt.tight_layout()
    plt.imshow(final_test_np[i].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predit image: {}".format(predictions[0]))
plt.show()

