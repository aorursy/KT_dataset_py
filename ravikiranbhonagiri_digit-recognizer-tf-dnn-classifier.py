import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
df_train.head()
df_test.head()
df_train.columns
df_train.shape
#for col in df_train.columns:

#  print(" unique values in {} are \n {}".format(col, np.unique(df_train[col])))
from sklearn.model_selection import train_test_split



X = df_train.drop(columns=['label'])

y = df_train['label']

X = X/255 ## normalisation



X_train, X_test, y_train, y_test = train_test_split( X , y, test_size=0.25, random_state=54)
import tensorflow as tf



model_1 = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(784,)),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(10)

])



model_2 = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(784,)),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(10)

])



model_3 = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(784,)),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(10)

])
model_1.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model_2.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model_3.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model_1.fit(X_train, y_train, epochs=10)
model_2.fit(X_train, y_train, epochs=10)
model_3.fit(X_train, y_train, epochs=10)
test_loss, test_acc = model_1.evaluate(X_test,  y_test, verbose=2)



print('\nTest accuracy:', test_acc)
test_loss, test_acc = model_2.evaluate(X_test,  y_test, verbose=2)



print('\nTest accuracy:', test_acc)
test_loss, test_acc = model_3.evaluate(X_test,  y_test, verbose=2)



print('\nTest accuracy:', test_acc)
pred = model_2.predict(df_test)
print(type(pred))

print(pred)
labels = []

for arr in pred:

  labels.append(arr.argmax())
submission = pd.DataFrame()

submission["ImageId"] = range(1, len(labels) + 1)

submission["Label"] = labels
submission.to_csv("submission.csv", index=False)
submission_check = pd.read_csv("submission.csv")

submission_check.head()