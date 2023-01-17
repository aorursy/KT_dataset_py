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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
dataset= pd.read_csv("../input/digit-recognizer/train.csv")
print(dataset.shape)
dataset.head() #dataframe
trainx= dataset.drop(['label'],axis=1).values #drop label column
trainy= dataset['label'].values #taking label column
trainx[0] #1 pixel value
trainy[0]
sns.countplot(trainy)
testdata = pd.read_csv("../input/digit-recognizer/test.csv")
testdata.head()
test = testdata.values
test[0]
trainx = trainx.astype('float32')
trainy = trainy.astype('int32')
test= test.astype('float32')
plt.figure(figsize=(17,11))
x, y = 5,4
for i in range(20):  
    plt.subplot(y, x, i+1)
    plt.imshow(trainx[i].reshape((28,28)),interpolation='nearest')
plt.show()
trainx = trainx/255.0
test = test/255.0
trainx = trainx.reshape(trainx.shape[0],28,28,1)
test = test.reshape(test.shape[0],28,28,1)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
trainy = pd.get_dummies(trainy)
trainy
trainx, valx, trainy, valy = train_test_split(trainx, trainy, test_size = 0.1, random_state=42)
from keras.models import Sequential
from keras import models, layers
import keras
#Instantiate an empty model
model = Sequential()

# C1 Convolutional Layer
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,1), padding='same'))

# S2 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

# C3 Convolutional Layer
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

# S4 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# C5 Fully Connected Convolutional Layer
model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
#Flatten the CNN output so that we can connect it with fully connected layers
model.add(layers.Flatten())

# FC6 Fully Connected Layer
model.add(layers.Dense(84, activation='tanh'))

#Output Layer with softmax activation
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=["accuracy"])
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
my_callbacks = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=4, verbose=1,factor=0.5,min_lr=0.00001)
datagen = ImageDataGenerator(
        rotation_range=15, 
        zoom_range = 0.1,  
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=False,  
        vertical_flip=False) 
model.summary()
datagen.fit(trainx)
history = model.fit_generator(datagen.flow(trainx,trainy, batch_size=128),epochs = 40, 
                  validation_data = (valx,valy),verbose = 1,callbacks= my_callbacks)
final_loss, final_acc = model.evaluate(valx, valy, verbose=0)
print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss function for RMSprop')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy for RMSprop')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
predictions = model.predict_classes(test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submit.csv", index=False, header=True)