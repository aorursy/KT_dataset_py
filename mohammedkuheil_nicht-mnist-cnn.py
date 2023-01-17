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

from tensorflow.keras import datasets,layers, models

from tensorflow import keras

from keras.layers import Dense, Dropout

from tensorflow.keras import regularizers

import matplotlib.pyplot as plt



print(tf.__version__)
train_data = pd.read_csv('../input/nicht-mnist/train.csv',header=None, index_col =0)

test_data = pd.read_csv('../input/nicht-mnist/test.csv',header=None , index_col = 0)
train_data.head()
test_data.tail()
y = train_data[1]

x = train_data.drop(columns=[1])
y.unique()
y.value_counts()
x = x / 255.0

test_data = test_data / 255.0
test_data.head()
x = x.values.reshape(-1,28,28,1)

test_data = test_data.values.reshape(-1,28,28,1)
test_data.shape
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()

y = pd.DataFrame(label_encoder.fit_transform(y))
from sklearn.model_selection import train_test_split

train_x, val_x,train_y, val_y = train_test_split(x, y, test_size = 0.2, random_state=1)
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001),input_shape=(28, 28,1),padding = 'same'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(Dropout(0.5))



model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1),padding = 'same'))

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001),padding = 'same'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(Dropout(0.5))



model.add(layers.Conv2D(256, (3, 3), activation='relu',padding = 'same'))



model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(layers.Dense(10))
model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])

EarlyStopp = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(train_x, train_y, epochs=100, batch_size=32,validation_data=(val_x, val_y),callbacks=[EarlyStopp])
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label='Training accuracy')

plt.plot(history.history['val_accuracy'], label = 'Validation accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label = 'Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()



test_loss, test_acc = model.evaluate(val_x,  val_y, verbose=2)

print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_data)

results = np.argmax(predictions,axis = 1)

results = pd.Series(results,name="target")
submission = pd.concat([pd.Series(range(0,9364),name = "Id"),results],axis = 1)

submission['target'] = label_encoder.inverse_transform(submission['target'])

submission.to_csv('nicht-mnist_cnn.csv', index=False)
submission