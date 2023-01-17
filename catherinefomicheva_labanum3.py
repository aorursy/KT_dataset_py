# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.metrics import classification_report, confusion_matrix

num_classes = 10
batch_size = 128
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

(trainX, trainY), (testX, testY) = cifar10.load_data()
trainY = to_categorical(trainY)
testY = to_categorical(testY)
train_images = trainX.astype('float32')
test_images = testX.astype('float32')
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.imshow(train_images[i])
plt.show()
model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape))
model1.add(Conv2D(64, (3,3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dense(num_classes, activation='softmax'))

model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
history1 = model1.fit(train_images, trainY, batch_size=batch_size, validation_data=(test_images, testY), epochs=50, verbose=1)
score = model1.evaluate(test_images, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.figure(figsize=plt.figaspect(0.5))
l = range(0, len(history1.history['val_loss']))
plt.plot(l, history1.history['val_loss'])
plt.title('val_loss')
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model2.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
model2.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model2.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model2.add(Dense(10, activation='softmax'))
#в качестве метода оптимизации буду использовать SGD, он показал себя лучше в процессе проведения тестов
lrate = 0.01
sgd = SGD(lr=lrate, momentum=0.9)
model2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history2 = model2.fit(train_images, trainY, batch_size=batch_size, validation_data=(test_images, testY), epochs=50, verbose=1)
score = model2.evaluate(test_images, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.figure(figsize=plt.figaspect(0.5))
l = range(0, len(history2.history['val_loss']))
plt.plot(l, history2.history['val_loss'])
plt.title('val_loss')
model3 = Sequential()
model3.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model3.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model3.add(Dropout(0.25))
model3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model3.add(Dropout(0.3))
model3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.3))
model3.add(Flatten())
model3.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model3.add(Dropout(0.4))
model3.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model3.add(Dropout(0.5))
model3.add(Dense(10, activation='softmax'))

lrate = 0.01
sgd = SGD(lr=lrate, momentum=0.9)
model3.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history3 = model3.fit(train_images, trainY, batch_size=batch_size, validation_data=(test_images, testY), epochs=50, verbose=1)
score = model3.evaluate(test_images, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.figure(figsize=plt.figaspect(0.5))
l = range(0, len(history3.history['val_loss']))
plt.plot(l, history3.history['val_loss'])
plt.title('val_loss')
Y_pred = model3.predict_classes(test_images)
y_cm = [np.argmax(testY[i]) for i in range(0,testY.shape[0])]
print(confusion_matrix(y_cm, Y_pred))
print(classification_report(y_cm, Y_pred))