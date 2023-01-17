# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from keras.layers import Conv2D, activations, Dropout,Flatten, MaxPooling2D, Dense, GlobalAveragePooling2D
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras import applications
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import cv2
import os
print(os.listdir("../input"))

%matplotlib inline
# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
train_data.head()
labels = train_data["label"]
train_data = train_data.drop(["label"], axis = 1)
train_data.head()
f, axr = plt.subplots(2,2)
img1 = np.asarray(train_data.iloc[3]).reshape(28,28)
img2 = np.asarray(train_data.iloc[4]).reshape(28,28)
img3 = np.asarray(train_data.iloc[6]).reshape(28,28)
img4 = np.asarray(train_data.iloc[9]).reshape(28,28)

axr[0,0].imshow(img1, cmap = plt.get_cmap("gray"))
axr[0,1].imshow(img2, cmap = plt.get_cmap("gray"))
axr[1,0].imshow(img3, cmap = plt.get_cmap("gray"))
axr[1,1].imshow(img4, cmap = plt.get_cmap("gray"))
train_data = np.asarray(train_data)
print("Shape of training data = ", train_data.shape)
test_data = np.asarray(test_data)
print("Shape of testing data = ", test_data.shape)
# pros_train_data = train_data / train_data.mean()
pros_train_data = train_data - np.mean(train_data) / np.std(train_data)
pros_test_data = test_data - np.mean(test_data) / np.std(test_data)
                                   
train_x, val_x, train_y, val_y = train_test_split(pros_train_data, labels, shuffle = True, test_size = 0.1)
print("Shape of training data = ",train_x.shape, train_y.shape)
print("Shape of validation data = ", val_x.shape, val_y.shape)
x_train = (train_x).reshape(-1, 28, 28, 1)
x_val = (val_x).reshape(-1, 28, 28, 1)

print(x_train.shape, x_val.shape)
train_y = np_utils.to_categorical(train_y)
val_y = np_utils.to_categorical(val_y)
print(train_y.shape, val_y.shape)

clf = Sequential()
clf.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (28,28,1), padding = "same"))
clf.add(MaxPooling2D((3,3), padding = "same"))
clf.add(Conv2D(32, kernel_size = (3,3), activation= "relu", padding="same"))
clf.add(MaxPooling2D((3,3), padding = "same"))
clf.add(Conv2D(64, kernel_size = (3,3), activation= "relu", padding="same"))
clf.add(Conv2D(128, kernel_size = (3,3), activation= "relu", padding="same"))
clf.add(MaxPooling2D((3,3), padding = "same"))
clf.add(Flatten())
clf.add(Dense(64, activation = "relu"))
clf.add(BatchNormalization())
clf.add(Dropout(0.5))
clf.add(Dense(10, activation = "softmax"))
clf.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
clf.summary()



history = clf.fit(x_train, train_y, batch_size=32,epochs=50,verbose=1,validation_data=(x_val, val_y))
x_test = np.asarray(pros_test_data).reshape(-1, 28,28, 1)
pred = clf.predict(x_test)

predictions = np.argmax(pred,axis=1)
predictions.shape

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("my_attempt.csv", index=False, header=True)
history.history.keys()

val_loss = history.history["val_loss"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
acc = history.history["acc"]

epochs = [i for i in range(len(loss))]

plt.plot(val_acc)
plt.plot(acc)
plt.title("model accuracy")
plt.xlabel("accuracy")
plt.ylabel("epochs")
plt.legend(["val_acc","acc"])
plt.show()


plt.plot(val_loss)
plt.plot(loss)
plt.title("model loss")
plt.xlabel("loss")
plt.ylabel("epochs")
plt.legend(["val_loss","loss"])
plt.show()


datagen =ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)

batches = datagen.flow(x_train, train_y, batch_size = 32)
val_batches = datagen.flow(x_val, val_y, batch_size = 32)

history=clf.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                    validation_data=val_batches, validation_steps=val_batches.n)

pred = clf.predict(x_test)
prediction = np.argmax(pred,axis=1)
prediction.shape

submission=pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)),
                         "Label": prediction})
submission.to_csv("my_third_best_submission2.csv", index=False, header=True)