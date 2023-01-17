import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, MaxPool2D, Dense
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
train.dropna(inplace = True)
print(train.shape)

train.head()
print(test.shape)
test.dropna(inplace = True)
print(test.shape)

test.head()
X_train = train.drop('label', axis = 1)
y_train = train.label

_ = sns.countplot(y_train)
X_train = X_train / 255.0
test = test / 255.0
if (str(type(X_train))[8:21] != 'numpy.ndarray'):
    X_train = X_train.values
if (str(type(test))[8:21] != 'numpy.ndarray'):    
    test = test.values


X_train = X_train.reshape(-1,28,28,1)
test = test.reshape(-1,28,28,1)

print('Convert Completed')
y_train = to_categorical(y_train, num_classes = 10)
train_X , val_X, train_y, val_y = train_test_split(
                                        X_train, y_train, 
                                        random_state = 2, 
                                        test_size = 0.1)
ixx = random.randint(0, len(train_X))
_ = plt.imshow(train_X[ixx][:, :, 0])
model = Sequential([
    
    Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28,28,1)),
    Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'),
    Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2,2), strides=(2,2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
    
])

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(
                                monitor='val_acc',
                                period=3,
                                fator=0.5,
                                min_lr = 0.00001
)
data_aug = ImageDataGenerator(
    rotation_range = 10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

data_aug.fit(train_X)
batch_size = 86
epochs = 30

history = model.fit_generator(data_aug.flow(train_X,train_y, batch_size=batch_size),
                              epochs = epochs, validation_data = (val_X,val_y),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction]
                              )
results = model.predict(test)
results = np.argmax(results, axis=1)

submission = pd.DataFrame({'ImageId': range(1,len(test)+1), 'Label': results})
submission.to_csv('image_submission.csv', index=False)
submission.head()