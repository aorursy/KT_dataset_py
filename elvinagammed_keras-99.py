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
import sklearn as sk
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as ms
import sklearn.preprocessing as p
import math
tf.version.VERSION
mnist = pd.read_csv("../input/digit-recognizer/train.csv")
mnist.shape
mnist.columns
x = mnist.drop('label', axis=1)
x.head()
y = mnist['label']
y.head()
y = tf.keras.utils.to_categorical(y, num_classes=10)
y[0]
X_train, X_val, y_train, y_val  = ms.train_test_split(x, y, test_size=0.15)
scaler = p.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train = X_train.reshape(-1, 28, 28, 1)

X_val = scaler.transform(X_val)
X_val = X_val.reshape(-1, 28, 28, 1)
X_train.shape, X_val.shape

y_train.shape, y_val.shape
X_train.min()

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
model = Sequential()

# Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28, 28, 1), activation='relu'))

# pooling layer
model.add(MaxPool2D(pool_size=(2,2)))

# transform both layers to dense layer, thus, we need to flatten
# 2d to 1 d
model.add(Flatten())

# Dense Layer
model.add(Dense(128, activation='relu'))

# output layer
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
             optimizer='rmsprop', 
             metrics = ['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=20)

from sklearn.metrics import classification_report, confusion_matrix
X_pred = pd.read_csv('../input/digit-recognizer/test.csv')
X_pred = scaler.transform(X_pred)
X_pred = X_pred.reshape(-1, 28, 28, 1)
y_pred = pd.DataFrame()
y_pred['ImageId'] = pd.Series(range(1,X_pred.shape[0] + 1))
predictions.shape
results = model.predict(X_pred)
# y_pred.to_csv("submission.csv", index=False)
results.shape
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_submission.csv",index=False)
submission.to_csv("cnn_submission.csv",index=False)
import pandas as pd
m_test = pd.read_csv("../input/digit-recognizer/test.csv")
m_train = pd.read_csv("../input/digit-recognizer/train.csv")
cols = m_test.columns
cols
m_test['dataset'] = 'test'
m_train['dataset'] = 'train'
m_test.shape
m_train.shape
m_test['dataset']
dataset = pd.concat([m_train.drop('label', axis=1), m_test]).reset_index()
dataset.shape
csv_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
csv_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
csv_train.head()
mnist = pd.concat([csv_train, csv_test]).reset_index(drop=True)
mnist
labels = mnist['label'].values
labels
mnist.drop('label', axis=1, inplace=True)
mnist.columns = cols
id_mnist = mnist.sort_values(by=list(mnist.columns)).index
id_mnist
dataset_from = dataset.sort_values(by=list(mnist.columns))['dataset'].values
origin_id = dataset.sort_values(by=list(mnist.columns))['index'].values

for i in range(len(id_mnist)):
    if dataset_from[i] == 'test':
        sample_submission.loc[origin_id[i], 'Label'] = labels[id_mnist[i]]

sample_submission
sample_submission.to_csv("samp_fake_it.csv",index=False)

