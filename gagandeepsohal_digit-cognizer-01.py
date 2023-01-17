# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.preprocessing import image

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

print(train.shape)

train.head(5)



test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print(test.shape)

test.head(5)
X_train = train.iloc[:,1:].values.astype('float32')

y_train = train.iloc[:,0].values.astype('int32')

X_test = test.values.astype('float32')

X_train = X_train.reshape(X_train.shape[0],28,28,1)

print(X_train.shape)

X_test = X_test.reshape(X_test.shape[0],28,28,1)

print(X_test.shape)
mean = X_train.mean().astype(np.float32)

st = X_train.std().astype(np.float32)



def standard(x):

    return(x-mean)/st
gen = image.ImageDataGenerator()



X = X_train

y = y_train

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

batches = gen.flow(X_train, y_train, batch_size=64)

val_batches = gen.flow(X_val, y_val, batch_size=64)
def model_made():

    model = tf.keras.models.Sequential([

        tf.keras.layers.Lambda(standard, input_shape=(28,28,1)),

        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),

        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),

        tf.keras.layers.MaxPooling2D(),   

        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),

        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation="relu"),

        tf.keras.layers.Dense(10, activation="softmax")

    ])

    

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    

    return model


model = model_made()

history = model.fit_generator(generator=batches, epochs=5, validation_data=val_batches)

results = model.predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

print(results)
submission=pd.DataFrame({"ImageId": list(range(1,len(results)+1)),

                         "Label": results})

submission.to_csv('set.csv', index=False, header=True)



submission
for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))