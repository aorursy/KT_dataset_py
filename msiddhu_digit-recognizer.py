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
import tensorflow as tf

data=pd.read_csv('../input/digit-recognizer/train.csv')

test=pd.read_csv('../input/digit-recognizer/test.csv')

import matplotlib.pyplot as plt

from keras.utils import to_categorical



X_train = data.iloc[:,1:]

y_train = data.iloc[:,0]

y_train = to_categorical(y_train, 10)





X_train = X_train.values.reshape(-1, 28, 28, 1)/255.

test = test.values.reshape(-1, 28, 28, 1)/255.
X_train.shape
data
from sklearn.model_selection import train_test_split

print(X_train.shape,y_train.shape)
trainData,testData, trainLabels, testLabels=train_test_split(X_train,y_train,test_size=0.2, random_state=0)
print(trainData.shape,trainLabels.shape,testLabels.shape,testData.shape)


	

def ModelConv(width, height, depth, classes):

    # initialize the model

	model = Sequential()

	inputShape = (height, width, depth)

	# first set of CONV => RELU => POOL layers

	model.add(Conv2D(32, (5, 5), padding="same",

	input_shape=inputShape))

	model.add(Activation("relu"))

	model.add(MaxPooling2D(pool_size=(2, 2)))

		# second set of CONV => RELU => POOL layers

	model.add(Conv2D(32, (3, 3), padding="same"))

	model.add(Activation("relu"))

	model.add(MaxPooling2D(pool_size=(2, 2)))

		# first set of FC => RELU layers

	model.add(Flatten())

	model.add(Dense(64))

	model.add(Activation("relu"))

	model.add(Dropout(0.5))

		# second set of FC => RELU layers

	model.add(Dense(64))

	model.add(Activation("relu"))

	model.add(Dropout(0.5))

		# softmax classifier

	model.add(Dense(classes))

	model.add(Activation("softmax"))

		# return the constructed network architecture

	return model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation,MaxPooling2D,Dense,Conv2D,Flatten,Dropout

INIT_LR = 1e-3

EPOCHS = 30

BS = 32

opt = Adam(lr=INIT_LR)

model=ModelConv(28,28,1,10)



model.compile(loss="categorical_crossentropy", optimizer=opt,

	metrics=["accuracy"])

model.summary()
history = model.fit(

	trainData, trainLabels,

	validation_data=(testData, testLabels),

	batch_size=BS,

	epochs=EPOCHS,

	verbose=1)
loss = history.history['loss']

val_loss = history.history['val_loss']

plt.plot(loss)

plt.plot(val_loss)
results = model.predict(test)


results = np.argmax(results, axis=1)

results = pd.Series(results, name='Label')

submission = pd.concat([pd.Series(range(1,28001), name='ImageID'), results], axis=1)

submission.to_csv('submission.csv', index=False)