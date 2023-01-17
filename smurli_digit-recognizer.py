# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

gen = image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)

print("Reading Data Set")
trainCsv = pd.read_csv("../input/train.csv")
testCsv = pd.read_csv("../input/test.csv")

X = (trainCsv.iloc[:,1:].values).astype('float32') # all pixel values
X = X.reshape(X.shape[0], 28, 28,1)
Y = trainCsv.iloc[:,0].values.astype('int32') # only labels i.e targets digits
#Convert Y to one-hot encoding
Y = to_categorical(Y)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.10, random_state=42)
batches = gen.flow(X_train, Y_train, batch_size=64,seed=40)
val_batches=gen.flow(X_val, Y_val, batch_size=64,seed=41)

X_test = testCsv.values.astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px

print("Done")
import random

print("Training Set:", trainCsv.shape)
print("Test Entry:", testCsv.shape)

count=0
for loc in random.sample(range(0, trainCsv.shape[0]), 3):
    img=X_train[loc]
    img=img.reshape((28,28))
    count+=1
    plt.subplot(130+count)
    plt.imshow(img,cmap='gray')
    plt.title(Y_train[loc])
count=0
plt.show()
for loc in random.sample(range(0, testCsv.shape[0]), 3):
    img=X_test[loc]
    img=img.reshape((28,28))
    count+=1
    plt.subplot(130+count)
    plt.imshow(img,cmap='gray')
from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model
import matplotlib.image as mpimg
from PIL import Image

model= Sequential()
model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
print("input shape ",model.input_shape)
print("output shape ",model.output_shape)

model.summary()
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#img=mpimg.imread('model_plot.png')
#imgplot = plt.imshow(img,interpolation='nearest')
#plt.show()

model.compile(optimizer=RMSprop(lr=0.001),
 loss='categorical_crossentropy',
 metrics=['accuracy'])

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=5, 
                    validation_data=val_batches, validation_steps=val_batches.n)

predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)
submissions.head()