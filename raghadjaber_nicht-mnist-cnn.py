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

from tensorflow.keras import layers
Nicht_train = pd.read_csv('../input/nicht-mnist/train.csv',header=None, index_col =0)

Nicht_test = pd.read_csv('../input/nicht-mnist/test.csv',header=None , index_col = 0)
Nicht_train.head()
y = Nicht_train.pop(1)

x = Nicht_train
x = x / 255.0

Nicht_test = Nicht_test / 255.0
x = x.values.reshape(-1,28,28,1)

Nicht_test = Nicht_test.values.reshape(-1,28,28,1)
y = y.values
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()

y = pd.DataFrame(label_encoder.fit_transform(y))
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state=1)
from keras.layers import Dense, Dropout
cnn = keras.models.Sequential()

cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1))) 

cnn.add(keras.layers.MaxPool2D(pool_size = 2, strides=2))

cnn.add(keras.layers.Flatten())

cnn.add(keras.layers.Dense(128, activation='relu'))

cnn.add(keras.layers.Dense(10,activation='softmax'))
cnn.compile(  optimizer= 'Adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = cnn.fit(x_train, y_train,

                  validation_data=(x_val, y_val), 

                  epochs=20)
test_loss, test_acc = cnn.evaluate(x_val,  y_val, verbose=2)

print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([cnn, 

                                         tf.keras.layers.Softmax()])
def plot_history(history, ep):

    import matplotlib.pyplot as plt

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']



    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs_range = range(ep)



    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)

    plt.plot(epochs_range, acc, label='Training Accuracy')

    plt.plot(epochs_range, val_acc, label='Validation Accuracy')

    plt.legend(loc='lower right')

    plt.title('Training and Validation Accuracy')



    plt.subplot(1, 2, 2)

    plt.plot(epochs_range, loss, label='Training Loss')

    plt.plot(epochs_range, val_loss, label='Validation Loss')

    plt.legend(loc='upper right')

    plt.title('Training and Validation Loss')

    plt.show()
plot_history(history, 20)
predictions = probability_model.predict(Nicht_test)
results = np.argmax(predictions,axis = 1)
results = pd.Series(results,name="target")
submission = pd.concat([pd.Series(range(0,9364),name = "Id"),results],axis = 1)
submission['target'] = label_encoder.inverse_transform(submission['target'])
submission.head()
submission.to_csv('sub_ensemble_10_cnn.csv', index=False)