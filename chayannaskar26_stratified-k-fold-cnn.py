import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



import tensorflow as tf
train = pd.read_csv('../input/train.csv')

train.head()
train.describe()
test = pd.read_csv("../input/test.csv")

test.head()
test.describe()
train.melt(id_vars="label")['value'].isnull().sum()
test.melt()['value'].isnull().sum()
train['label'].value_counts().sort_index()
# Plot

plt.figure(figsize=(8, 4))

sns.set_style("whitegrid")

sns.countplot(x="label", data=train)

plt.xlabel("Label")

plt.ylabel("Count")

plt.show()
y = train['label']

X = train.drop(['label'], axis = 1)
X = X / 255

test = test / 255
y = np.array(y)
X = X.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
plt.imshow(X[0][:,:,0])
from sklearn.model_selection import StratifiedKFold
# Tensorflow Keras CNN Model

model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu", input_shape = X.shape[1:]))

model.add(tf.keras.layers.MaxPool2D(2,2))



model.add(tf.keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu"))

model.add(tf.keras.layers.MaxPool2D(2,2))



model.add(tf.keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu"))

model.add(tf.keras.layers.MaxPool2D(2,2))



model.add(tf.keras.layers.Conv2D(256, (3,3), padding = "same", activation = "relu"))

model.add(tf.keras.layers.MaxPool2D(2,2))



model.add(tf.keras.layers.Flatten())



model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy', 

              metrics=['accuracy'])
model.summary()
# Stratified K-Fold

k_fold = StratifiedKFold(n_splits=12, random_state=12, shuffle=True)



for k_train_index, k_test_index in k_fold.split(X, y):

    model.fit(X[k_train_index,:], y[k_train_index], epochs=5)
val_loss, val_acc = model.evaluate(X, y)

val_acc
model.save("dr_cnn_model.h5")
test_pred = model.predict(test)
submission = pd.DataFrame()

submission['ImageId'] = range(1, (len(test)+1))

submission['Label'] = np.argmax(test_pred, axis=1)
submission.head()
submission.shape
submission.to_csv("submission.csv", index=False)