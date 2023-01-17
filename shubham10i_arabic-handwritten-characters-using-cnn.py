import numpy as np
import os
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
train_images = pd.read_csv('../input/ahcd1/csvTrainImages 13440x1024.csv')
train_label = pd.read_csv('../input/ahcd1/csvTrainLabel 13440x1.csv')

test_images = pd.read_csv('../input/ahcd1/csvTestImages 3360x1024.csv')
test_label = pd.read_csv('../input/ahcd1/csvTestLabel 3360x1.csv')
train_images = train_images.iloc[:,:].values
train_label = train_label.iloc[:,:].values

test_images = test_images.iloc[:,:].values
test_label = test_label.iloc[:,:].values
train_images[0].shape
train_label[0]
train_images.shape
train_images = train_images.reshape(train_images.shape[0], 32, 32)
test_images = test_images.reshape(test_images.shape[0],32,32)
print("Train images shape: ",train_images.shape)
print("Train labels shape: ",train_label.shape)

print("Test images shape: ",test_images.shape)
print("Test labels shape: ",test_label.shape)
fig, ax = plt.subplots(4,2,figsize=(20,20))

for i in range(4):
    for j in range(2):
        ax[i,j].imshow(train_images[i])
train_images.max()
train_images.min()
train_label.max()
#Normalize 

train_images = train_images/255.0
test_images = test_images/255.0
train_images.max()
len(train_images)
X_train = train_images[1000:]
X_valid = train_images[:1000]

y_train = train_label[1000:]
y_valid = train_label[:1000]
num_classes = len(np.unique(train_label)) + 1
num_classes
X_test = test_images[:]
y_test = test_label[:]
y_train = to_categorical(y_train,num_classes)
y_valid = to_categorical(y_valid,num_classes)
y_test = to_categorical(y_test,num_classes)
y_train
print("X train shape: ",X_train.shape)
print("Training samples: ",X_train.shape[0])
print("Validation samples: ",X_valid.shape[0])
print("Testing samples: ",X_test.shape[0])
X_train = X_train.reshape([-1, 32, 32, 1])
X_test = X_test.reshape([-1, 32, 32, 1])
X_valid = X_valid.reshape([-1, 32, 32, 1])
X_train.shape
model = Sequential()
X_train.shape[1:]
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32,32,1)))
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.summary()
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(29,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])
data_gen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)
data_gen.fit(X_train)
batch_size=128

history = model.fit_generator(data_gen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    epochs=100, verbose=2,
                    validation_data=(X_valid, y_valid),
                    validation_steps=X_valid.shape[0] // batch_size)
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
sns.lineplot(x=np.arange(1, 101), y=history.history.get('loss'), ax=ax[0, 0])
sns.lineplot(x=np.arange(1, 101), y=history.history.get('accuracy'), ax=ax[0, 1])
sns.lineplot(x=np.arange(1, 101), y=history.history.get('val_loss'), ax=ax[1, 0])
sns.lineplot(x=np.arange(1, 101), y=history.history.get('val_accuracy'), ax=ax[1, 1])
ax[0, 0].set_title('Training Loss vs Epochs')
ax[0, 1].set_title('Training Accuracy vs Epochs')
ax[1, 0].set_title('Validation Loss vs Epochs')
ax[1, 1].set_title('Validation Accuracy vs Epochs')
fig.suptitle('Basic CNN model', size=16)
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1]*100)
print('\n', 'Total test Loss:', score[0]*100)
