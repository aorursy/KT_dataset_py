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
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("Tensorflow version " + tf.__version__)
np.random.seed(10)
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
training_csv_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
submission_test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
training_csv_data.info()
train_data, test_data = train_test_split(training_csv_data,test_size=0.2,random_state=10)
print(train_data.info())
print(test_data.info())

y_train = train_data["label"]
x_train = train_data.drop(["label"],axis=1)
y_test = test_data["label"]
x_test = test_data.drop(["label"],axis=1)

print(x_train.loc[3698,:].values.tolist())
y_train.head()
print(x_train.shape)
print(y_train.shape)
x_train_norm = x_train/255
x_test_norm = x_test/255
x_train = x_train_norm
x_test = x_test
x_train = x_train.values
y_train = y_train.values
x_test = x_test.values
y_test = y_test.values
# x_train =x_train.reshape(-1,28,28)
# x_test =x_test.reshape(-1,28,28)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(784,)))
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
model.summary()
# history = model.fit(x_train, y_train, epochs=10)
model.fit(x_train,y_train,epochs = 10)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis =1)
y_test
count = 0
for v in range(0,len(y_test)):

    if y_test[v] != y_pred[v]:
        count=count+1
print(count)
results = model.predict(submission_test_data)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("anmol_submission.csv",index=False)

submission.head()