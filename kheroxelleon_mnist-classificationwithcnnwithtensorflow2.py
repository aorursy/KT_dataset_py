import tensorflow as tf
tf.__version__
import pandas as pd

import numpy as np
df_train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
from tensorflow.keras.utils import to_categorical
df_train.head()
X=df_train.drop('label',axis=1)/255

Y=to_categorical(df_train['label'])
X=np.array(X).reshape(42000,28,28,1)

X_predict=np.array(df_test/255).reshape(28000,28,28,1)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu",input_shape=(28,28,1)))

model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu",input_shape=(28,28,1)))

model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu",input_shape=(28,28,1)))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(256,activation="relu"))

model.add(tf.keras.layers.Dense(512,activation="relu"))

model.add(tf.keras.layers.Dense(256,activation="relu"))

model.add(tf.keras.layers.Dense(10,activation="softmax"))



model.summary()
from tensorflow.keras.optimizers import RMSprop
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.0001),metrics=['acc'])
model.fit(X_train,Y_train,epochs=30,batch_size=256)
model.evaluate(X_test,Y_test,batch_size=128)