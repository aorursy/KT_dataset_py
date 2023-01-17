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

from tensorflow import keras

import numpy as np

import random

from sklearn.model_selection import train_test_split
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
y=train['label']

y.head()
X=train.drop('label',axis=1)

X.head()
X=X/255

test=test/255
print(X.shape)
X= X.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
print(X.shape)
y = np.array(y)

print(y.shape)
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1)
model = keras.Sequential([

    tf.keras.layers.Conv2D(16, (3,3),padding="same", activation = 'relu', input_shape = X_train.shape[1:]),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3),padding="same", activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3),padding="same", activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10)

])
model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = model.fit(X_train, y_train, verbose=1, epochs=6, batch_size=128, validation_data=(X_val, y_val))
test_labels=model.predict(test)

test_labels
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
plotLearningCurve(history,6)
test_labels.shape
print(history)
results = np.argmax(test_labels,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



print(submission)



submission.to_csv("submission.csv",index=False)
results.head()