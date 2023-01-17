import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
y_train = train['label']
X_train = train.drop('label', axis=1)

del train #To free more space
sns.countplot(y_train)
y_train.value_counts()
print('Corrupted images in train: ',X_train.isnull().any().any())
print('Corrupted images in test: ',test.isnull().any().any())
X_train = X_train / 255.0
test = test / 255.0

#Normalising of the data to a [0 - 1] range. CNN converges faster than on [0 - 255]
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

#Reshaping to 3D, the images are 28 x 28 x 1. Seeing as they are black and white, they only use 1 channel
# One hot encoding. For example, the value one would equal [0,1,0,0,0,0,0,0,0,0]

y_train = to_categorical(y_train, num_classes = 10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)

#The distribtion of y labels is fairly equal as seen in the graph earlier. Therefore, a random split of 10% validation and 90% training won't cause
#certain labels to be over represented.
plt.imshow(X_train[42][:,:,0])
X_val.shape
#Basic model
# [Conv2D -> Relu -> Conv2D -> Relu -> MaxPool2D -> Dropout -> Flatten -> Dense -> Dropout -> Out]

model1 = Sequential()

model1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

model1.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model1.add(MaxPool2D(pool_size=(2, 2)))

model1.add(Dropout(0.25))

model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(10, activation='softmax'))
model1.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
history1 = model1.fit(X_train, y_train,
          batch_size=128, epochs=30,
          verbose=2,
          validation_data=(X_val, y_val))

print('\n')
print('Best training score: ', max(history1.history['accuracy']))
print('Best validation score: ', max(history1.history['val_accuracy']))

# Plotting the results
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.tight_layout()

fig
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)
history1_aug = model1.fit_generator(datagen.flow(X_train, y_train,
          batch_size=128), epochs=30,
          verbose=2,
          validation_data=(X_val, y_val))

print('\n')
print('Best training score: ', max(history1_aug.history['accuracy']))
print('Best validation score: ', max(history1_aug.history['val_accuracy']))

# Plotting the results
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history1_aug.history['accuracy'])
plt.plot(history1_aug.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history1_aug.history['loss'])
plt.plot(history1_aug.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.tight_layout()

fig
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*3 -> Flatten -> Dense -> Dropout -> Out
#Credit to "https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6"

model2 = Sequential()

model2.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model2.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model2.add(MaxPool2D(pool_size=(2,2)))
model2.add(Dropout(0.25))


model2.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model2.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model2.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model2.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model2.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model2.add(Dropout(0.25))


model2.add(Flatten())
model2.add(Dense(256, activation = "relu"))
model2.add(Dropout(0.5))
model2.add(Dense(10, activation = "softmax"))
model2.compile(optimizer = 'RMSprop' , loss = "categorical_crossentropy", metrics=["accuracy"])
history2 = model2.fit(X_train, y_train,
          batch_size=128, epochs=30,
          verbose=2,
          validation_data=(X_val, y_val))

print('\n')
print('Best training score: ', max(history2.history['accuracy']))
print('Best validation score: ', max(history2.history['val_accuracy']))

# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.tight_layout()

fig
history2_aug = model2.fit_generator(datagen.flow(X_train, y_train,
          batch_size=128), epochs=30,
          verbose=2,
          validation_data=(X_val, y_val))

print('\n')
print('Best training score: ', max(history2_aug.history['accuracy']))
print('Best validation score: ', max(history2_aug.history['val_accuracy']))

# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history2_aug.history['accuracy'])
plt.plot(history2_aug.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history2_aug.history['loss'])
plt.plot(history2_aug.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.tight_layout()

fig
#Comparing the models with and without data augmentation

print('Model 1: Best Train accuracy={0:.5f}, Best Validation accuracy={1:.5f}'.format( max(history1.history['accuracy']), max(history1.history['val_accuracy']) ) )
print('Model 2: Best Train accuracy={0:.5f}, Best Validation accuracy={1:.5f}'.format(max(history2.history['accuracy']), max(history2.history['val_accuracy'])))

print('\nModel 1 with data augmentation: Best Train accuracy={0:.5f}, Best Validation accuracy={1:.5f}'.format(max(history1_aug.history['accuracy']), 
                                                                                                             max(history1_aug.history['val_accuracy'])))

print('Model 2: with data augmentation: Best Train accuracy={0:.5f}, Best Validation accuracy={1:.5f}'.format(max(history2_aug.history['accuracy']), 
                                                                                                              max(history2_aug.history['val_accuracy'])))

results = model1.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("MNIST-CNN-NoAug-RMS.csv", index=False)