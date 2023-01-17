import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout

from tensorflow.keras.utils import to_categorical   
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
Y = np.array(train["label"])

print(Y.shape)

Y = to_categorical(Y)

print(Y.shape)
del train["label"]
X = train.values
img = X[14]

img = np.array(img, dtype = 'float')

img = img.reshape(28, 28)



plt.imshow(img)

plt.show()
X.shape
X = X.reshape(X.shape[0], 28, 28, 1)
X.shape
model1 = Sequential()

model2 = Sequential()
cnn1 = Conv2D(32, (3,3), padding = "same", activation = "relu", input_shape = (28, 28, 1))

pool1 = MaxPool2D(pool_size = (2,2))

cnn2 = Conv2D(64, (3,3), padding = "same", activation = "relu")

pool2 = MaxPool2D(pool_size = (2,2))

drop1 = Dropout(0.2)
flatten = Flatten()

dense1 = Dense(units = 128, activation = "relu")

drop2 = Dropout(0.1)

dense2 = Dense(units = 10, activation = "softmax")
model1.add(cnn1)

model1.add(pool1)

model1.add(cnn2)

model1.add(pool2)

model1.add(drop1)

model1.add(flatten)

model1.add(dense1)

model1.add(drop2)

model1.add(dense2)
model2.add(cnn1)

model2.add(pool1)

model2.add(cnn2)

model2.add(pool2)

model2.add(drop1)

model2.add(flatten)

model2.add(dense1)

model2.add(dense2)
model1.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics = ["accuracy"])
model2.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
# model1.fit(X, Y, epochs = 20, batch_size = 128)
model2.fit(X, Y, epochs = 5, batch_size = 128)
X_test = np.array(test.values)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test.shape
# predictions1 = model1.predict(X_test)

predictions2 = model2.predict(X_test)
# pred1 = np.argmax(predictions1, axis = 1)

pred2 = np.argmax(predictions2, axis = 1)
# s1 = pd.DataFrame(pred1)

s2 = pd.DataFrame(pred2)
labels = np.array([i for i in range(1, 28001)])
# del s1["ImageI"]

# s1.insert(0, "ImageId", labels, True)
s2.insert(0, "ImageId", labels, True)
# s1.rename(columns = {0:'Label'}, inplace = True)
s2.rename(columns = {0:'Label'}, inplace = True)
# s1.to_csv("submit1.csv", index=False)
s2.to_csv("submit2.csv", index=False)