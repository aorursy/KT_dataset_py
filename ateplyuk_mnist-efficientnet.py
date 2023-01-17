import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
!pip install git+https://github.com/qubvel/efficientnet
from efficientnet import EfficientNetB3
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten 

from keras.models import Model

from keras import optimizers

from keras.utils import np_utils

import cv2

import matplotlib.pyplot as plt

%matplotlib inline
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 
X_train.shape, test.shape
# Normilize data

X_train = X_train.astype('float32')

test = test.astype('float32')

X_train /= 255

test /= 255
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
X_train.shape, test.shape
X_train3 = np.full((42000, 28, 28, 3), 0.0)



for i, s in enumerate(X_train):

    X_train3[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) 
g = plt.imshow(X_train3[1])
test3 = np.full((28000, 28, 28, 3), 0.0)



for i, s in enumerate(test):

    test3[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) 
g = plt.imshow(test3[1])
X_train3.shape, test3.shape
Y_train = np_utils.to_categorical(Y_train, 10)

Y_train
model = EfficientNetB3(weights='imagenet', input_shape = (28,28,3), include_top=False)
model.trainable = False
x = model.output

x = Flatten()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(units = 10, activation="softmax")(x)

model_f = Model(input = model.input, output = predictions)

model_f.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy'])
%%time

# Train model

history = model_f.fit(X_train3, Y_train,

              epochs=10,

              batch_size = 128,

              validation_split=0.1,

              shuffle=True,

              verbose=2)
import json



with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
%%time

# Prediction

test_predictions = model_f.predict(test3)
test_predictions.shape
test_predictions[0]
# select the index with the maximum probability

results = np.argmax(test_predictions,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)
submission.head()