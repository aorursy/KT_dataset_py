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

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import tensorflow.keras as keras



from sklearn.model_selection import train_test_split



%matplotlib inline
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)



g_tr = sns.countplot(Y_train)



Y_train.value_counts()
X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 10)
np.random.seed(2)
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
print(X_train.shape)

print(X_val.shape)

print(Y_train.shape)

print(Y_val.shape)
model=keras.models.Sequential([keras.layers.Conv2D(32,3,padding = 'same', activation='relu', input_shape=[28,28,1]),

                               keras.layers.Conv2D(32, 3,padding = 'same', activation='relu'),

                               keras.layers.MaxPooling2D(pool_size=2),

                               keras.layers.Dropout(.25),

                               keras.layers.Conv2D(64, (3, 3),padding = 'same', activation='relu'),

                               keras.layers.Conv2D(64, (3, 3),padding = 'same', activation='relu'),

                               keras.layers.MaxPooling2D(pool_size=2, strides=2),

                               keras.layers.Dropout(.25),

                               keras.layers.Flatten(),

                               keras.layers.Dense(256, activation='relu'),

                               keras.layers.Dropout(.5),

                               keras.layers.Dense(10, activation='softmax'),

])
model.compile(optimizer = 'RMSprop' , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 25

batch_size = 86
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 

                    validation_data = (X_val, Y_val), verbose = 2)
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission_final.csv",index=False)