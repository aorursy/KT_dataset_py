import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau

import os
print(os.listdir("../input"))

batch_size = 128
num_classes = 10
img_rows, img_cols = 28, 28
np.random.seed(7)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

x_train = train.iloc[:,1:785]
y_train = train.iloc[:,0]
x_test = test
x_train = x_train.values
x_test = x_test.values 

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)


model_2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_rows, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax'),])

from keras.optimizers import RMSprop
model_2.compile(optimizer=RMSprop(lr=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = model_2.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=12,
          verbose=1,
          validation_split=0.3)

fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')


predictions = model_2.predict_classes(x_test, verbose=0)
submissions = pd.DataFrame({"ImageId" : list(range(1, len(predictions)+1)), "Label" : predictions.astype(int)})
submissions.to_csv("digit_recognizer.csv", index=False, header=True)
