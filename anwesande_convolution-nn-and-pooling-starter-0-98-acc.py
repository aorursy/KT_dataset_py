# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras import regularizers

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

df.columns
y_train=df.label.values

df=df.drop(["label"],axis=1)

df.head()
x_train=df.values

x_train.shape

x_train_shaped=x_train.reshape(42000,28,28,1)
x_test=df_test.values

x_test.shape

x_test_shaped=x_test.reshape(28000,28,28,1)
model=tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64,(3,3),activation="relu",input_shape=(28,28,1)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64,activation="relu"),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10,activation="softmax")

    

])
model.compile(metrics=["accuracy"], loss="sparse_categorical_crossentropy",optimizer="adam")
model.fit(x_train_shaped,y_train,epochs=30)
pred=model.predict(x_test_shaped)

pred.shape
results = np.argmax(pred,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("result.csv",index=False)