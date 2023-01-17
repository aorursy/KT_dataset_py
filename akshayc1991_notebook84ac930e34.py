import csv
import pandas as pd
import numpy as np
filename = "../input/creditcardfraud/creditcard.csv"

df = pd.read_csv(filename)
df.head()
features[:5]

features = df.drop("Class",axis=1)
features = features.values
target = df["Class"]
target = target.values

target.shape
val_rate = int(len(features)*.2)
train_features = features[:-val_rate]
train_target = target[:-val_rate]
val_features = features[-val_rate:]
val_target = target[-val_rate:]
print(train_features.shape)
print(train_target.shape)
print(val_features.shape)
print(val_target.shape)
counts = np.bincount(train_target[:,])
print(counts[0])
print(counts[1])
print(len(train_target))
mean = np.mean(train_features,axis=0)
train_features -=mean
val_features -=mean
std = np.std(train_features,axis=0)
train_features /=std
val_features /=std
train_features.shape[-1]
import tensorflow as tf
from tensorflow import keras

input_layer = tf.keras.layers.Input(shape=(train_features.shape[1],))
x = tf.keras.layers.Dense(256,activation="relu")(input_layer)
x = tf.keras.layers.Dense(256,activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256,activation="relu")(x)
output_layer = tf.keras.layers.Dense(1,activation = "sigmoid")(x)
model = tf.keras.Model(inputs=input_layer,outputs=output_layer)

model.summary()
metrics = [
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=metrics)
weights_for_0 = 1/counts[0]
weights_for_1 = 1/counts[1]
weights_for_0
weights_for_1
class_weight= {0:weights_for_0,1:weights_for_1}

history = model.fit(train_features,train_target,batch_size=256,epochs=30,verbose=2,validation_data=(val_features, val_target),class_weight=class_weight)
