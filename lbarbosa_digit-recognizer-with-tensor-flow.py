import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
print(train.shape) # 42000 rows, 785 columns [one is the label]
print(test.shape) # 28000 rows, 784 columns [unlabeled]
y=train.label
y=y.values
train_nl=train.drop('label', axis=1)
train_nl=train_nl.values
train_nl=train_nl/255
X_train, X_test, y_train, y_test = train_test_split(train_nl, y, test_size=0.1)
# Building the tf.keras model by stacking layers. Selecting an optimizer and loss function used for training
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=6)

model.evaluate(X_test, y_test) # Model is 97.8% accurate
# preping values from test.csv file for prediction and submission
test=test.values
test=test/255
y_pred = model.predict_classes(test)
y_pred = pd.DataFrame(y_pred)
y_pred["ImageId"]=y_pred.index+1
y_pred = y_pred.rename(columns = { 0:"Label"})
y_pred = y_pred[["ImageId", "Label"]]
y_pred.head
y_pred.to_csv("submission.csv")