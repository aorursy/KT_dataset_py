import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD

import os
print(os.listdir("../input"))

# load the data
train = pd.read_csv('../input/train.csv')
labels = train.iloc[:,0].values.astype('int32')
X_train = train.iloc[:,1:].values.astype('float32')
X_test = pd.read_csv('../input/test.csv').values.astype('float32')

# reshape the images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# scaling
scale = np.max(X_train)
X_train /= scale
X_test /= scale

# apply mean subtraction to the data
mean = np.mean(X_train)
X_train -= mean
X_test -= mean

# convert the labels from integers to vectors
lb = LabelBinarizer()
Y_train = lb.fit_transform(labels)

print("Check Data")
print(X_train[0].shape)
print(Y_train.shape)
print("Max: {}".format(scale))
print("mean: {}".format(mean))
#visualize an image
fig = plt.figure()
plt.imshow(X_train[10][:,:,0], cmap='gray', interpolation='none')
plt.xticks([])
plt.yticks([])
# build the model

model = Sequential()
inputShape = (28, 28, 1)
classes = Y_train.shape[1]
INIT_LR = 5e-3

opt = SGD(lr=INIT_LR, momentum=0.9)

# define the first (and only) CONV => RELU layer
model.add(Conv2D(32,(3, 3),padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(classes))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# fit the model to the training set
model.fit(X_train, Y_train, epochs=100,verbose=1, validation_split=0.1)
# generate predictions
predictions = model.predict_classes(X_test, verbose=0)
print(predictions)
pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions}).to_csv("submission.csv", index=False, header=True)