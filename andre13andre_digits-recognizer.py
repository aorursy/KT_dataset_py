# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



train_images=train.drop('label',axis=1)

train_label=train['label']
test=np.array(test).reshape(28000,28,28,1)

train_images=np.array(train_images).reshape(42000,28,28,1)

train_images=train_images/255

test=test/255
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.998):

            print("\nReached 99.9% accuracy so cancelling training!")

            self.model.stop_training = True



  # Your Code



callbacks = myCallback()
import tensorflow as tf

model = tf.keras.models.Sequential([

    # YOUR CODE STARTS HERE

    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),

    tf.keras.layers.MaxPool2D((2,2)),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPool2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256,activation='relu'),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(10,activation='softmax')

    # YOUR CODE ENDS HERE

])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

model.fit(train_images,train_label,epochs=25,callbacks=[callbacks])
predict=model.predict_classes(test)

submission=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission['Label']=predict

submission.to_csv('submit.csv',index=False)