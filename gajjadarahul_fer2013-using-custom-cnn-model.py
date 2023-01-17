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
data = pd.read_csv('/kaggle/input/fer2013/fer2013.csv')
data.head(3)
data.Usage.unique()
data = data.drop(['Usage'], axis=1)
data.shape
from sklearn.model_selection import train_test_split
train = data['pixels']
test = data['emotion']
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.1, random_state=10)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
width, height = 48, 48
X_train1 = []
for i in X_train:
    X_train1.append([int(p) for p in i.split()])
X_train1 = np.array(X_train1)/255.
X_train1.shape
X_test1 = []
for i in X_test:
    X_test1.append([int(p) for p in i.split()])
X_test1 = np.array(X_test1)/255.
X_test1.shape
X_test1[:2]
X_train1[:4]
X_train1 = X_train1.reshape(X_train1.shape[0], 48, 48, 1)
X_train1.shape
X_test1 = X_test1.reshape(X_test1.shape[0], 48, 48, 1)
X_test1.shape
X_train1.dtype, X_test1.dtype
X_train32 = X_train1.astype('float32')
X_test32 = X_test1.astype('float32')
X_train32.dtype, X_test32.dtype
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Flatten, Dense, Activation, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation, BatchNormalization
model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (5,5), input_shape = (48, 48, 1), padding='same'))
model.add(Conv2D(64, kernel_size=(5,5), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same'))
model.add(Conv2D(128, kernel_size=(5,5), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
model.add(Conv2D(256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
from tensorflow.keras.utils import to_categorical
n_epochs = 20
batch_size = 64
lr = 0.0001
history = model.fit(X_train32, to_categorical(y_train), 
                    batch_size = batch_size, epochs = n_epochs, validation_data= (X_test32, to_categorical(y_test)))
pd.DataFrame(history.history).tail()
y_pred = model.predict_classes(X_test32)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(to_categorical(y_test), axis=1), y_pred)
cm
model.save('CNN_model.h5')

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=".",
                  name="CNN_model.pb",
                  as_text=False)
