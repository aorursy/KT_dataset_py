# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
y_train = train_data["label"]
x_train = train_data.drop(labels = ["label"], axis = 1) 
test_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
x_test = train_data.drop(labels = ["label"], axis = 1) 
x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)
# As we have 784 for each image, we can reshape to 28x28
import matplotlib.pyplot as plt
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_train[i,:,:,0], 'gray')
    plt.title(y_train.iloc[i])
from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes = 10)
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
batch_size = 128
epochs = 100


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"))
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
opt = adam(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
es_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.3,
          shuffle=True, callbacks=[es_callback])

# predicted class
num_rows = 6
num_cols = 15
sample_size = num_rows * num_cols
indices = np.arange(sample_size)
x_pred = x_test[indices,:,:]
predictions = model.predict(x_pred)
x_pred = np.squeeze(x_test[indices,:,:])
y_pred = np.argmax(predictions,axis=1)

num_images = num_rows*num_cols
plt.figure(figsize=(num_cols*2, num_rows*2))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plt.imshow(x_pred[i])
    plt.title(y_pred[i])
plt.show()

# predict results
results = model.predict(x_test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("MNIST-submission.csv",index=False)
