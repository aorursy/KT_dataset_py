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
ls
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img=mpimg.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/NORMAL2-IM-0373-0001.jpeg')

imgplot = plt.imshow(img)

# example of progressively loading images from file
from keras.preprocessing.image import ImageDataGenerator
# create generator
datagen = ImageDataGenerator(
zoom_range=[0.5,1.0],brightness_range=[0.2,1.0],horizontal_flip=True,rescale=1./255
)
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/train/', class_mode='binary',batch_size=32,target_size=(224, 224))
val_it = datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/val/', class_mode='binary',batch_size=32,target_size=(224, 224))
test_it = datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/test/', class_mode='binary',batch_size=32,target_size=(224, 224))
# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
from keras.applications import VGG16
model = VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3))

#Freeze layers
for layer in model.layers[:]:
  layer.trainable= False

import keras
from keras import Sequential
from keras.layers import Dense,Flatten,Dropout

new_model = Sequential()

new_model.add(model)

new_model.add(Flatten())
new_model.add(Dense(1024,activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(1, activation='sigmoid'))



new_model.summary()
new_model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(lr=1e-4),metrics=['acc'])
his = new_model.fit(train_it,epochs=30,validation_data=val_it)
# Plot the accuracy and loss curves
acc = his.history['acc']
val_acc = his.history['val_acc']
loss = his.history['loss']
val_loss = his.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = new_model.evaluate(test_it)
print('test loss, test acc:', results)

