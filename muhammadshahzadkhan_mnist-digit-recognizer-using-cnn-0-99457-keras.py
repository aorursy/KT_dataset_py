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
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data.shape
data.columns
input_data = data.drop('label', axis=1)
input_data.shape
output_data = data['label']
output_data.shape
input_data.head(1)
max(input_data.iloc[0])
from sklearn import preprocessing
MinMaxScaler = preprocessing.MinMaxScaler()
input_data = MinMaxScaler.fit_transform(input_data)
#Reshape to required tensor shape in Keras
input_data = input_data.reshape(-1, 28, 28, 1)
input_data.shape
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_data = MinMaxScaler.fit_transform(test)
test_data = test_data.reshape(-1, 28, 28, 1)
test_data.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data.values, test_size=0.3, random_state=1)
#To convert target data (One hot encoding #preprocessing)
# 1 ==> [1 0 0 0 0 0 0 0 0 0]
# 2 ==> [0 1 0 0 0 0 0 0 0 0]
# 3 ==> [0 0 1 0 0 0 0 0 0 0]
from keras.utils.np_utils import to_categorical
y_test.shape
y_test
X_train.shape
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
y_test_cat.shape
y_train_cat.shape
y_train_cat
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, AveragePooling2D
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(#rescale = 1./255,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rotation_range = 20,
                               shear_range = 0.3,
                               zoom_range = 0.3,
                               horizontal_flip = True)
generator.fit(X_train)
K.clear_session()

model = Sequential()

#model.add(generator())

model.add(Conv2D(32, (3, 3), activation='tanh', input_shape=(28, 28, 1), padding="SAME"))
model.add(Conv2D(32, (3, 3), activation='tanh', padding="SAME"))
#model.add(Conv2D(32, (3, 3), activation='tanh', padding="SAME"))
model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

#model.add(Conv2D(128, (3, 3), activation='tanh'))
model.add(Conv2D(64, (3, 3), activation='tanh', padding="SAME"))
model.add(Conv2D(64, (3, 3), activation='tanh', padding="SAME"))
model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='tanh', padding="SAME"))
model.add(Conv2D(128, (3, 3), activation='tanh', padding="SAME"))
model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(784, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adamax',
              metrics=['accuracy'])
model.summary()
# Reduce learning by measuring "validation accuracy"
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
#Now training model through augmentation
#For augmentation please follow the link "https://keras.io/api/preprocessing/image/"
K.clear_session()
hh = model.fit(generator.flow(X_train, y_train_cat, batch_size=64), validation_data=(X_test, y_test_cat),
          steps_per_epoch=len(X_train) / 64, epochs=30, verbose=1, callbacks=[learning_rate_reduction])
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(hh.history['accuracy'])
plt.plot(hh.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')
# store the weights of trained model on augmented images data
trained_weights = model.get_weights()
learning_rate_reduction_2 = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.3, 
                                            min_lr=0.00001)
K.clear_session()

# to the previous trained model weights
model.set_weights(trained_weights)
#Now training on actual data and initial weights are adjusted to already traing model on augmented data
h = model.fit(X_train, y_train_cat,
                      batch_size = 64,
                      validation_data=(X_test, y_test_cat),
                      epochs=50,
                      verbose=1, callbacks=[learning_rate_reduction_2])
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')
model.evaluate(X_test, y_test_cat)
predictions = model.predict(test_data)
predictions.shape
pred =  np.argmax(predictions, axis = 1)
pred.shape
sample = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
output = pd.DataFrame({'ImageId': sample.ImageId, 'Label': pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")