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

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from keras.utils import to_categorical

from sklearn.model_selection import KFold

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPool2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.optimizers import SGD
# load train and test dataset

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test  = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print(train.shape, test.shape)

train.head()
# X and Y

X = train.drop("label", axis =1)

y = train["label"]

X_train = X.to_numpy().reshape(X.shape[0], 28,28, 1)

y_train = to_categorical(y)



#Scaling Pixels

X_train = X_train.astype("float32")

X_train = X_train/255.0
#define model

cnn = Sequential()

cnn.add( Conv2D( filters = 32, kernel_size = 3, activation = "relu", input_shape = [28,28,1]) )

cnn.add( MaxPool2D( pool_size = 2 , strides =2) )

cnn.add( Conv2D( filters = 32, kernel_size = 3, activation = "relu" ) )

cnn.add( MaxPool2D( pool_size = 2 , strides =2) )

cnn.add(Flatten())

cnn.add (Dense( units = 128, activation = "relu" )) #Hidden Layer

cnn.add(Dense (units = 10, activation ="softmax")) # Binary classification: 1 or 0

opt = SGD(lr=0.01, momentum=0.9) # optimizer for compiling

cnn.compile( optimizer =opt , loss = "categorical_crossentropy", metrics = ["accuracy"])
#Evaluate Model

scores, histories = list(), list()



# prepare cross validation

n_folds=5

kfold = KFold(n_folds, shuffle=True, random_state=1)



# enumerate splits

dataX,dataY = X_train,y_train

for train_ix, test_ix in kfold.split(dataX):

    # define model

    model = cnn

    # select rows for train and test

    trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]

    # fit model

    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)

    # evaluate model

    _, acc = model.evaluate(testX, testY, verbose=0)

    print('> %.3f' % (acc * 100.0))

    # stores scores

    scores.append(acc)

    histories.append(history)
# plot diagnostic learning curves

for i in range(len(histories)):

    # plot loss

    plt.subplot(2, 1, 1)

    plt.title('Cross Entropy Loss')

    plt.plot(histories[i].history['loss'], color='red', label='train')

    plt.plot(histories[i].history['val_loss'], color='black', label='test')

    # plot accuracy

    plt.subplot(2, 1, 2)

    plt.title('Classification Accuracy')

    plt.plot(histories[i].history['accuracy'], color='red', label='train')

    plt.plot(histories[i].history['val_accuracy'], color='black', label='test')

plt.show()
#Fit the model

cnn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
#predicting and final submission

test = test.to_numpy().reshape(test.shape[0], 28,28, 1)

test = test.astype("float32")

test = test/255.0

preds = cnn.predict(test)



final = pd.DataFrame(np.argmax(preds, axis=1), columns=['Label'])

final.insert(0, 'ImageId', final.index + 1)

final.to_csv('submission.csv', index=False)