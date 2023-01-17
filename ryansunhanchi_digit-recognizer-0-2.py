# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten,Convolution2D,MaxPooling2D, BatchNormalization
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
#ATTRIBUTES
M=42000
n=28000 #num of test images

#load training data
labeled_images=pd.read_csv('../input/digit-recognizer/train.csv')
X=(labeled_images.iloc[0:M,1:].values).astype('float32')/255
y=to_categorical(labeled_images.iloc[0:M,0])

#load test data
test_images=pd.read_csv('../input/digit-recognizer/test.csv')
X_test=test_images.values.astype('float32')/255
#mnist dataset
#X=(np.load('../input/mnist-train/mnist_images.npy')/255).astype('float32')
#y=to_categorical(np.load('../input/mnist-train/mnist_labels.npy'))
print (X.shape)
print (X.dtype)
print (y.shape)
X=X.reshape(X.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)
#X=X>(100/255)
#X_test=X_test>(100/255)
print (X.shape)
print (y.shape)
print (X_test.shape)
gen=ImageDataGenerator(rotation_range=7, width_shift_range=0.05, shear_range=0.1,
                               height_shift_range=0.05, zoom_range=0.05)
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.02)
batches = gen.flow(X_train, y_train, batch_size=64)
dev_batches=ImageDataGenerator().flow(X_dev, y_dev, batch_size=64)
print (X_train.shape)
print (y_train.shape)
print (X_dev.shape)
print (y_dev.shape)
def get_cnn_model():
    model = Sequential([
        Convolution2D(16,(3,3),padding='same', activation='relu',input_shape=(28,28,1)), #28*28*16
        BatchNormalization(axis=1),
        Convolution2D(32,(3,3),padding='same', activation='relu'),#28*28*32
        MaxPooling2D(),#14*14*64
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),#12*12*64
        BatchNormalization(axis=1),
        Convolution2D(128,(3,3), activation='relu'),#10*10*128
        MaxPooling2D(),#5*5*128
        #Dropout(0.25),
        Flatten(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        #Dropout(0.25),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(lr=0.001,decay=0.005), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
model= get_cnn_model()
history=model.fit_generator(generator=batches, steps_per_epoch=int(batches.n), epochs=3, 
                    validation_data=dev_batches, validation_steps=dev_batches.n)
accuracy= model.evaluate(X_dev,y_dev)
print (accuracy)
prediction=model.predict_classes(X_dev, verbose=0)
print (prediction.shape)
#indicate where the prediction is incorrect
indices=np.argwhere(prediction!=np.argmax(y_dev,axis=1))
print (indices)
i=0
for index in indices[:9]:
    plt.subplot(330 + (i+1))
    i+=1
    plt.imshow(X_dev[index].reshape(28,28), cmap=plt.get_cmap('gray'))
    plt.title(str(np.argmax(y_dev[index]))+" not "+str(int(prediction[index])))
predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)
