import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import ndimage
import matplotlib.pyplot as plt
import time

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data.head()
data_as_np = data.to_numpy()
labels = data_as_np[:,0].reshape((42000,1)) #To avoid having issues with arrays of np shape (42000,)
pixels = data_as_np[:,1:].reshape((42000,28,28,1)) / 255 # normalizing the inputs

labels, pixels = shuffle(labels, pixels)

r = int(np.random.uniform(0,41999))
digit=pixels[r,:,:,0] * 255 #pyplot expects a 2D array for a gray scale image
plt.imshow(digit, cmap=plt.cm.gray)   
X_train, X_CV, y_train, y_CV = train_test_split(pixels, labels, test_size = 0.2 )

print(f'Training data: \nX_train shape : {X_train.shape} and y_train shape : {y_train.shape}\n')
print(f'Cross validation data: \nX_CV shape : {X_CV.shape} and y_CV shape : {y_CV.shape}')
inputs = keras.Input(shape = (28,28,1))

conv1 = layers.Conv2D(6, (5,5), strides = (1,1), activation = 'tanh')
x = conv1(inputs)

x = layers.AveragePooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(16, (5,5), strides = (1,1), activation = 'tanh')(x)
x = layers.AveragePooling2D(pool_size=(2,2))(x)

x = layers.Flatten()(x)
x = layers.Dense(120, activation = 'tanh')(x)
x = layers.Dense(84, activation = 'tanh')(x)

outputs = layers.Dense(10, activation = 'softmax')(x)

model = keras.Model(inputs = inputs, outputs = outputs, name="MNISTv1")

model.summary()
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
             optimizer = keras.optimizers.Adam(),
             metrics = ['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=10)
test = model.evaluate(X_CV, y_CV, verbose=2)
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_as_np = test.to_numpy()

pixels_test = test_as_np.reshape((28000,28,28,1)) / 255

tic = time.time()
result = model.predict(pixels_test, batch_size=32)
toc = time.time()

elapsed = (toc-tic)*1000
print(f'The inference time for the 28 000 examples batch size is {elapsed} ms')


result = result.argmax(axis=1).reshape((28000,1))
i = int(np.random.uniform(0,27999))

tic = time.time()
inference = model.predict(pixels_test[i:i+1,:,:,:])
toc = time.time()

elapsed = (toc-tic)*1000

print(f'The result is {inference.argmax(axis=1)} and the inference time is {elapsed} ms')

#print(np.average(pixels_test[i:i+1,:,:,:]))
#print('this is the shit \n',np.average(pixels_test[i,:,:,0])) #Theses two lines can be used to make sure we are looking at the same example.

digit=pixels_test[i,:,:,0] * 255 
plt.imshow(digit, cmap=plt.cm.gray)  
a = np.arange(1,28001).reshape((28000,1))
test_result = np.concatenate((a, result), axis = 1)

print(test_result)
np.savetxt('test.csv', test_result, fmt='%i', delimiter=',', header="ImageId,Label", comments='')
check = pd.read_csv('/kaggle/working/test.csv')
check.head(10)