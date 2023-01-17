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
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(800, input_dim = 784))

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="SGD")
model.summary()
X_train = X_train / 255
model.fit(X_train, Y_train)
test = pd.read_csv("../input/test.csv")

test.head()
test = test / 255
result = model.predict(test)
result
res = np.argmax(result, axis=1)

res
df = pd.DataFrame({"ImageID":range(1,28001), "Label":res})

df.head()
df.to_csv("sub.csv", index=False)