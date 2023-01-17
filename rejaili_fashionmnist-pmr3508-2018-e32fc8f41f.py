import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import keras
import matplotlib.pyplot as plt
train_pure = np.load("../input/train_images_pure.npy")
print(train_pure.shape)
train_noisy = np.load("../input/train_images_noisy.npy")
train_rot = np.load("../input/train_images_rotated.npy")
train_both = np.load("../input/train_images_both.npy")
test = np.load("../input/Test_images.npy")
labels = pd.read_csv('../input/train_labels.csv').label.values
plt.figure(figsize=[20,7])
for i in np.unique(labels):
    plt.subplot(3,10,1+i)
    plt.imshow(train_pure[np.where(labels==i)[0][0]],cmap='gray')
    plt.subplot(3,10,11+i)
    plt.imshow(train_rot[np.where(labels==i)[0][0]],cmap='gray')
    plt.subplot(3,10,21+i)
    plt.imshow(train_noisy[np.where(labels==i)[0][0]],cmap='gray')
print(np.unique(labels))
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 128
num_classes = len(np.unique(labels))
epochs = 50

img_rows, img_cols = train_pure.shape[1], train_pure.shape[2]

x_train = train_pure.reshape(train_pure.shape[0],train_pure.shape[1],train_pure.shape[2],1)
x_noisy = train_noisy.reshape(train_noisy.shape[0],train_noisy.shape[1],train_noisy.shape[2],1)
x_rot = train_rot.reshape(train_rot.shape[0],train_rot.shape[1],train_rot.shape[2],1)
x_both = train_both.reshape(train_both.shape[0],train_both.shape[1],train_both.shape[2],1)
x_test = test.reshape(test.shape[0],test.shape[1],test.shape[2],1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'amostras de treino')
print(x_test.shape[0], 'amostras de teste')

y_train = keras.utils.to_categorical(labels, num_classes)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=50,
          verbose=1)
model.evaluate(x_rot, y_train)
model.evaluate(x_noisy, y_train)
model.evaluate(x_both, y_train)
from scipy.signal import medfilt
plt.subplot(1,2,1)
plt.imshow(test[10],cmap='gray')
plt.subplot(1,2,2)
plt.imshow(medfilt(test[10]),cmap='gray')
train_filt = np.zeros(train_noisy.shape)
for i in range(train_noisy.shape[0]):
    train_filt[i,:,:] = medfilt(train_noisy[i,:,:])
x_filt = train_filt.reshape(train_filt.shape[0],train_filt.shape[1],train_filt.shape[2],1)
model.evaluate(x_filt, y_train)
train_both_filt = np.zeros(train_both.shape)
for i in range(train_both.shape[0]):
    train_both_filt[i,:,:] = medfilt(train_both[i,:,:])
x_both_filt = train_both_filt.reshape(train_both_filt.shape[0],train_both_filt.shape[1],train_both_filt.shape[2],1)
model.evaluate(x_both_filt, y_train)
test_filt = np.zeros(test.shape)
for i in range(test.shape[0]):
    test_filt[i,:,:] = medfilt(test[i,:,:])
x_test_filt = test_filt.reshape(test_filt.shape[0],test_filt.shape[1],test_filt.shape[2],1)
prediction = pd.read_csv("../input/sample_sub.csv", index_col=0)
prediction.label = np.argmax(model.predict(x_test_filt), axis=1)
prediction.to_csv("submission.csv")
model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(28,28,1)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(Dropout(0.4))
model2.add(Flatten())
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(num_classes, activation='softmax'))
model2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model2.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=50,
          verbose=1)
model2.evaluate(x_rot, y_train)
model2.evaluate(x_noisy, y_train)
model2.evaluate(x_filt, y_train)
model2.evaluate(x_both, y_train)
model2.evaluate(x_both_filt, y_train)
prediction.label = np.argmax(model2.predict(x_test_filt), axis=1)
prediction.to_csv("submission2.csv")
model2.fit(x_both_filt, y_train,
          batch_size=batch_size,
          epochs=50,
          verbose=1)
model2.evaluate(x_rot, y_train)
model2.evaluate(x_noisy, y_train)
model2.evaluate(x_filt, y_train)
model2.evaluate(x_both, y_train)
model2.evaluate(x_both_filt, y_train)
prediction.label = np.argmax(model2.predict(x_test_filt), axis=1)
prediction.to_csv("submission3.csv")
np.unique(prediction.label)
