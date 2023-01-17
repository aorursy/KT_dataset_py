import tensorflow.keras as keras

import glob

import os as os

import cv2 as cv2

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from tensorflow.keras import optimizers



from sklearn.model_selection import train_test_split



%matplotlib inline

%xmode plain

sns.set()
path = '../input/chest_xray/chest_xray/train/'
os.listdir(path)
DIAGNOSIS = ['NORMAL/', 'PNEUMONIA/']
normal_xrays = np.array([cv2.resize(cv2.imread(img,  cv2.IMREAD_GRAYSCALE), (720, 720) )  for img in glob.glob(path+DIAGNOSIS[0]+"*.jpeg") ])

pneunomic_xrays = np.array([cv2.resize(cv2.imread(img,  cv2.IMREAD_GRAYSCALE), (720, 720) )  for img in glob.glob(path+DIAGNOSIS[1]+"*.jpeg")])
numbers = np.array([len(normal_xrays), len(pneunomic_xrays)])
sns.barplot(np.arange(2), numbers)
fig = plt.figure()

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222, sharex = ax1, sharey = ax1)

ax3 = fig.add_subplot(223, sharex = ax1, sharey = ax1)

ax4 = fig.add_subplot(224, sharex =ax1, sharey = ax1)



ax1.imshow(normal_xrays[0])

ax2.imshow(normal_xrays[1])

ax3.imshow(normal_xrays[2])

ax4.imshow(normal_xrays[3])



fig.suptitle('X-rays of normal patients', fontsize=20)
fig = plt.figure()

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222, sharex = ax1, sharey = ax1)

ax3 = fig.add_subplot(223, sharex = ax1, sharey = ax1)

ax4 = fig.add_subplot(224, sharex =ax1, sharey = ax1)



ax1.imshow(pneunomic_xrays[0])

ax2.imshow(pneunomic_xrays[1])

ax3.imshow(pneunomic_xrays[2])

ax4.imshow(pneunomic_xrays[3])



fig.suptitle('X-rays of pneumonic patients', fontsize=20)
fdata = np.vstack((normal_xrays, pneunomic_xrays))

del normal_xrays

del pneunomic_xrays
labels = np.vstack((np.zeros((numbers[0],1)), np.ones((numbers[1],1))))
X_train, X_test, y_train, y_test = train_test_split(fdata, labels, test_size=0.3, random_state=42)

X_train = X_train.reshape(-1, 720, 720, 1)

X_test = X_test.reshape(-1, 720, 720, 1)

del fdata
model = keras.Sequential()



model.add(keras.layers.Conv2D(64, kernel_size=(5,5) , strides=(3,3), activation='relu', input_shape = X_train.shape[1:]))

model.add(keras.layers.MaxPool2D(pool_size=(2,2)))



model.add(keras.layers.Conv2D(64, kernel_size=(5,5), strides=(3,3), activation='relu'))

model.add(keras.layers.MaxPool2D(pool_size=(2,2)))



model.add(keras.layers.Flatten())



model.add(keras.layers.Dense(256, activation='relu'))

model.add(keras.layers.BatchNormalization())



model.add(keras.layers.Dense(256, activation='relu'))

model.add(keras.layers.Dense(1, activation='sigmoid'))

callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss',

                                  min_delta = 0,

                                  patience = 5,

                                  verbose = 0, 

                                  mode='auto')]
optim =  optimizers.SGD(lr = 0.0001, momentum = 0.0)

model.compile(loss='binary_crossentropy',

              optimizer=optim,

              metrics=['acc'])
model.summary()
history = model.fit(X_train, y_train, epochs=20, batch_size=16,

                    validation_data=(X_test, y_test), callbacks = callbacks,

                    )
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs  = range(0, len(acc))



fig = plt.figure(figsize=(16,10))



ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222 )



ax1.plot(acc, label = 'train set')

ax1.plot(val_acc, label = 'test set')

ax1.set_title('Accuracy')

ax1.legend()

ax2.plot(loss, label = 'train set')

ax2.plot(val_loss, label= 'test set')

ax2.legend()

ax2.set_title('Loss')