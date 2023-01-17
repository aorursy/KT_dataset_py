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
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head()
Y = train['label']
X = train.drop('label',axis=1)
X.shape
X = X/255
test = test/255
print(X.shape,test.shape)
X = X.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
Y = np.array(Y)
Y = Y.reshape(-1,1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
print(x_train.shape,x_test.shape)

print(y_train.shape,y_test.shape)
from keras.layers import *

import keras

import tensorflow as tf

model = keras.Sequential([

    tf.keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=X.shape[1:]),

    tf.keras.layers.Activation('relu'),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Dropout(0.25),



    tf.keras.layers.Conv2D(32, (3, 3), padding='same'),

    tf.keras.layers.Activation('relu'),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Dropout(0.25),



    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),

    tf.keras.layers.Activation('relu'),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512),

    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Dropout(0.5),

    keras.layers.Dense(10)

])
model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = model.fit(x_train, y_train, verbose=1, epochs=10, batch_size=128, validation_data=(x_test, y_test))
result = model.predict(test)
import matplotlib.pyplot as plt



def plotLearningCurve(history,epochs):

    epochRange = range(1,epochs+1)

    plt.plot(epochRange,history.history['accuracy'])

    plt.plot(epochRange,history.history['val_accuracy'])

    plt.title('Model Accuracy')

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.legend(['Train','Validation'],loc='upper left')

    plt.show()



    plt.plot(epochRange,history.history['loss'])

    plt.plot(epochRange,history.history['val_loss'])

    plt.title('Model Loss')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.legend(['Train','Validation'],loc='upper left')

    plt.show()
plotLearningCurve(history,10)
results = np.argmax(result,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



print(submission)



submission.to_csv("submission.csv",index=False)