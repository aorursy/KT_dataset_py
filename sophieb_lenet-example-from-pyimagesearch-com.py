# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head()
test.head()
# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
 
class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        # initialize the model
        model = Sequential()
        # first set of CONV => RELU => POOL
        model.add(Convolution2D(20, 5, 5, border_mode="same",
            input_shape=(height, width,depth)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
         # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
         # return the constructed network architecture
        return model
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import cv2
X = (train.iloc[:,1:].values).astype('float32') 
#We separate the target variable
y = train.iloc[:,0].values.astype('int32')
#We get the test data
test = test.values.astype('float32')
#We reshape the train set
X = X.reshape((X.shape[0], 28, 28))
X = X[:, :, :, np.newaxis]

#We reshape the test set
test = test.reshape((test.shape[0], 28, 28))
test = test[:, :, :, np.newaxis]
X.shape
#We split the train set
X_train, X_test, y_train, y_test=train_test_split(
    X / 255.0, y.astype("int"), test_size=0.33)
X_train.shape
#We transform the train and test labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10,
    weightsPath=None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
print("[INFO] training...")
model.fit(X_train,y_train, batch_size=128, nb_epoch=20,verbose=1)
 
# show the accuracy on the testing set
print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(X_test, y_test,batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
#We check some of our predictions
nrows=2
ncols=3
n=0
elems=np.random.choice(np.arange(0,test.shape[0]),nrows*ncols)
fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
for row in range(nrows):
    for col in range(ncols):
        probs = model.predict(test[np.newaxis,n])
        prediction = probs.argmax(axis=1)
        ax[row,col].imshow(test[n][:,:,0])
        ax[row,col].set_title("Predicted label :{}".format(prediction))
        n=n+1
