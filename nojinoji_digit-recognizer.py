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

import pandas as pd
os.getcwd()
train_df=pd.read_csv("../input/digit-recognizer/train.csv")
train_df.head()
train_df.info()
train_df.describe()
train_df.isnull().any().sum()
test_df=pd.read_csv("../input/digit-recognizer/test.csv")

test_df.isnull().any().sum()
train_y=train_df["label"]
train_y.head()
train_x=train_df.drop("label", axis=1)

train_x.head()
print(train_x.shape)

print(train_y.shape)
train_x=train_x.values.reshape(-1,28,28,1)

train_x=train_x/255

train_x.shape
train_y=train_y.values

train_y.shape
DESIRED_ACCURACY=0.999





class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('accuracy')>DESIRED_ACCURACY):

      print("\nReached 97% accuracy so cancelling training!")

      self.model.stop_training = True



callback = myCallback()
model=tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (5,5), padding="same", activation="relu", input_shape=(28, 28,1)),

    tf.keras.layers.Conv2D(32, (5,5), padding="same", activation="relu"),

    tf.keras.layers.MaxPooling2D((2,2), padding="same"),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(32, (5,5), padding="same", activation="relu"),

    tf.keras.layers.MaxPooling2D((2,2), padding="same"),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(20, activation="relu"),

    tf.keras.layers.Dense(10, activation="softmax")

])



model.summary()
'''

from tensorflow.keras.optimizers import RMSprop

model.compile(loss="binary_crossentropy",

             optimizer=RMSprop(lr=0.001),

             metrics=["accuracy"])

'''

from tensorflow.keras.optimizers import Adam 



model.compile(optimizer="adam",

              loss="sparse_categorical_crossentropy",

              metrics=['accuracy'])
history=model.fit(

    x=train_x,

    y=train_y,

    batch_size=80,

    steps_per_epoch=500,

    epochs=30,

    callbacks=[callback]

)
test_x=test_df

test_x=test_x/255
test_x.shape
test_x=test_x.values.reshape(-1,28,28,1)

test_x.shape
test_y=model.predict(test_x)

test_y[0]
import numpy as np

results=np.argmax(test_y,axis=1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submition.csv", index=False)