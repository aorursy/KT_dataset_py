import numpy as np 

import pandas as pd
train = pd.read_csv("../input/train.csv")
train.head()
train['pixel150'].head()
Y_train = train['label']

Y_train.head()
X_train = train.drop(labels=['label'], axis=1)

X_train.shape
X_train.head()
import matplotlib.pyplot as plt

tr = X_train.iloc[1].values.reshape(28,28)

plt.imshow(tr)
from keras.utils import np_utils



Y_train = np_utils.to_categorical(Y_train, 10)

Y_train
X_train = X_train / 255
X_train.shape
X_train = X_train.values.reshape(42000,28,28,1)

X_train.shape
plt.imshow(X_train[1][:,:,0])
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (28,28,1)))

model.add(Flatten())

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=['accuracy'])
model.summary()
%%time

model.fit(X_train, Y_train, epochs = 5)
test = pd.read_csv("../input/test.csv")

test.head()
test = test / 255
test = test.values.reshape(len(test),28,28,1)

test.shape
result = model.predict(test)
result
res = np.argmax(result, axis=1)

res
df = pd.DataFrame({"ImageID":range(1,28001), "Label":res})

df.head()
df.to_csv("sub.csv", index=False)