import pandas as pd
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Y = train['label']
X = train.drop(labels=["label"], axis=1)
X = X / 255.0
test = test / 255.0
from sklearn.model_selection import train_test_split
random_seed=0
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=random_seed)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
plt.figure()
Xshow = X_train.values.reshape(-1,28,28,1)
plt.imshow(Xshow[0][:,:,0])
plt.colorbar()
plt.grid(False)
model = models.Sequential()
model.add(layers.SeparableConv2D(32,(3,3), activation="elu", input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.SeparableConv2D(64,(3,3), activation="elu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.SeparableConv2D(64,(3,3), activation="elu"))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="elu"))
model.add(layers.Dense(10, activation="softmax"))
model.summary()
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
X_val = X_val.values.reshape(-1,28,28,1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
X = X.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)
Y = to_categorical(Y, num_classes = 10)
model.fit(X, Y, epochs=20)
Y_val = to_categorical(Y_val, num_classes = 10)
test_loss, test_acc = model.evaluate(X_val, Y_val)
print("Test Accuracy: {}".format(test_acc))
predictions = model.predict(test)
predictions = np.argmax(predictions, axis=1)
submission = pd.read_csv('../input/sample_submission.csv')
submission["Label"] = predictions
submission.head()
submission.to_csv('./simpleMNIST.csv', index=False)
