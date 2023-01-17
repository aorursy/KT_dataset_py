# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
test_df.head()
print(test_df.shape)
train_X = train_df.iloc[:,1:].values.astype("float32")
train_Y = train_df.iloc[:,0].values.astype("int")
test_X = test_df.values.astype("float32")
print(train_X.shape)
train_X = train_X /255.
test_X = test_X / 255.
train_X = train_X.reshape((train_X.shape[0],28,28))
for i in range(0,3):
    plt.subplot(130 + (i+1))
    plt.imshow(train_X[i],cmap='gray')
    plt.title(train_Y[i])
train_X = train_X.reshape((train_X.shape[0],28,28,1))
test_X = test_X.reshape((test_X.shape[0],28,28,1))
from keras.utils.np_utils import to_categorical
print(train_Y.shape)
train_Y = to_categorical(train_Y)
print(train_Y.shape)
from sklearn.model_selection import train_test_split

train_X, val_X,train_Y,val_Y = train_test_split(train_X, train_Y,test_size=0.1, random_state=42)
print(val_X.shape)
print(train_X.shape)
from keras.preprocessing import image
gen = image.ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.08, height_shift_range=0.08)
batch_size=64
train_batches = gen.flow(train_X, train_Y, batch_size=batch_size)
val_batches = gen.flow(val_X, val_Y, batch_size=batch_size)
from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
def create_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    BatchNormalization(axis=1)
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    model.add(Conv2D(48, kernel_size=3, activation='relu'))
    BatchNormalization(axis=1)
    model.add(Dropout(0.2))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    BatchNormalization(axis=1)
    model.add(Flatten())
    BatchNormalization(axis=1)
    model.add(Dense(512,activation='relu'))
    BatchNormalization(axis=1)
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
model = create_model()
model.summary()

history = model.fit_generator(generator=train_batches, steps_per_epoch=train_batches.n / batch_size, validation_data=val_batches, validation_steps=val_batches.n / batch_size, epochs=20)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
predictions = model.predict_classes(test_X, verbose=0)

plt.imshow(test_X[0][:,:,0])
print(predictions[0])
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)

