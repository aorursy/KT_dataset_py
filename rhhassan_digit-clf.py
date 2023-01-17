import tensorflow as tf

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
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
import pandas as pd
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.shape
y = train.pop('label')
X = train/255.0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.6)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#df_train.shape, df_test.shape
import keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential([
    keras.layers.Dense(512, input_shape=(784,), activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
from keras.optimizers import RMSprop
num_epochs = 25
with tpu_strategy.scope():
    model.compile(loss='sparse_categorical_crossentropy',optimizer='RMSprop', metrics=['accuracy'])
    history = model.fit(X_train,y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=0)

import matplotlib.pyplot as plt
%matplotlib inline

epochs = range(1, num_epochs + 1)
plt.figure(figsize=(15,4))
plt.subplot(121)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(epochs, history.history['loss'], 'b', label='train_loss')
plt.plot(epochs, history.history['val_loss'], 'bo', label='test_loss')
plt.legend(loc='lower right')

plt.subplot(122)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(epochs, history.history['accuracy'],'r',label='train_acc')
plt.plot(epochs,history.history['val_accuracy'], 'ro', label='test_acc')
plt.legend(loc='lower right')
plt.show()
help(to_csv)
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
result = model.predict(test)
a = []
for row in range(len(result)):
    a = np.append(a,np.argmax(result[row]))
a.shape
#from keras.utils.np_utils import to_categorical
#result = to_categorical(result, num_classes = 10)
import numpy as np
sub = pd.DataFrame({'ImageID' : pd.Series(range(1,28001)), 'Label' : a}).astype(int)
sub.to_csv('rh-submission.csv',index=False)
print(os.listdir())