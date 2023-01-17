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
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
import tensorflow as tf
X = df.drop(['label'], axis = 1)
Y = df['label']
Y
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y)
x_train
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.values.astype('float32')
y_train = y_train.values.astype('float32')
x_test = x_test.values.astype('float32')
y_test = y_test.values.astype('float32')

x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train
test = test / 255.0
test = test.values.astype('float32')
test = test.reshape(test.shape[0], 28, 28,1)
x_train.shape
class StopTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if (logs.get('accuracy') == 1):
            print ("\nReached 100% accuracy so cancelling training!")
            self.model.stop_training = True
            
callbacks = StopTrainingCallback()
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)    
])
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(x_train, y_train, epochs = 35, callbacks = [callbacks])
model.evaluate(x_test, y_test)
predictions = model.predict_classes(test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)
