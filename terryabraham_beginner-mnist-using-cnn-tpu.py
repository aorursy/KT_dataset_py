# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import keras

print("Tensorflow version " + tf.__version__)
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train
#So in y_train we input the label column values and change the type to float32.* astype float32 seems to be very important for TPU processing *

y_train=train['label'].values.astype('float32')

y_train
#we drop label columns in X_train

X_train=train.drop(labels='label',axis=1)

X_train
X_train.shape
X_train=X_train.astype('float32')

# normalize to range 0-1

X_train=X_train/255
X_train=X_train.values.reshape(42000,28,28,1)

X_train.shape
#1st image

plt.imshow(X_train[0][:,:,0])
#2nd image

plt.imshow(X_train[1][:,:,0])
plt.imshow(X_train[41999][:,:,0])
def new_model():

    model=tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss="sparse_categorical_crossentropy",optimizer="Adam",metrics=['accuracy'])

    return model
%%time

# instantiating the model in the strategy scope creates the model on the TPU

with strategy.scope():

    model=new_model()

model.fit(X_train,y_train,epochs=12,steps_per_epoch=40,verbose=2)
#load the test dataset

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test
#as we normalized training set we also normalize and reshape test data also

test=test.astype('float32')

test=test/255

test=test.values.reshape(len(test),28,28,1)

test.shape
%%time

prediction=model.predict(test)
# select the index with the maximum probability



results = np.argmax(prediction,axis = 1)

results = pd.Series(results,name="Label")
results
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)



submission.head()