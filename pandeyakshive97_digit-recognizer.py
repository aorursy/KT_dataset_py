import pandas as pd

import numpy as np
train_data = pd.read_csv("../input/train.csv")
print(train_data)
train_data = np.array(train_data)
print(train_data)
train_label = train_data[:, 0]
print(train_label.shape)
x_train = train_data[:, 1:]
print(x_train.shape)
test_data = pd.read_csv("../input/test.csv")

test_data = np.array(test_data)
print(test_data)
x_test = test_data[:,:]
print(x_test.shape)
from keras import models

from keras import layers
model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_shape=(784,)))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

x_train = x_train.astype('float32')/255

x_test = x_test.astype('float32')/255
from keras.utils import to_categorical
train_label = to_categorical(train_label)
model.fit(x_train, train_label, epochs=5, batch_size=128)
predictions = model.predict(x_test)
results = []

for prediction in predictions:

    idx = 0

    for i in range(len(prediction)):

        if prediction[i] > prediction[idx]:

            idx = i

    results.append(idx)
print(np.asarray(results).shape)

temp = []

for i in range(1, 28001):

    temp.append(i)

print(np.asarray(temp).shape)
from pandas import DataFrame

df_dict = {'ImageId':temp, 'Label':results}

df = DataFrame(df_dict, columns=['ImageId', 'Label'])

print(df)

filename = 'digit_prediction.csv'

df.to_csv(filename,index=False)