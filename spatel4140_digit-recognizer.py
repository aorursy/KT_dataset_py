# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

train.head()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(np.array(train.iloc[:, 1:]), 

                                                    np.array(train["label"]), 

                                                    test_size=.2, random_state=0)

print(x_train.shape)

print(x_test.shape)
x_train = x_train.reshape(-1, 28, 28)

x_test = x_test.reshape(-1, 28, 28)
from matplotlib import pyplot as plt



plt.imshow(x_train[0])

plt.show()
x_train_normalized = x_train.astype(np.float32)/255

x_test_normalized = x_test.astype(np.float32)/255
from keras.utils import np_utils



y_train_categorical = np_utils.to_categorical(y_train, 10)

y_test_categorical = np_utils.to_categorical(y_test, 10)

y_train_categorical[0]
from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout



model = Sequential()

model.add(Dense(32, activation="relu", input_shape=(28, 28)))

model.add(Dense(128, activation="relu"))

#model.add(Dropout(.2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

#model.add(Dropout(.5))

model.add(Dense(10, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])



np.random.seed(0)

model.fit(x_train_normalized, y_train_categorical, epochs=15)
model.evaluate(x_test_normalized, y_test_categorical)[1]
incorrect_test = (np.argmax(model.predict(x_test_normalized), axis=1) != y_test)

"{0} of {1} wrong".format(incorrect_test.sum(), len(y_test))
plt.imshow(x_test[incorrect_test][0])

plt.show()



print("   true: {0}".format(y_test[incorrect_test][0]))

print("predict: {0}".format(np.argmax(model.predict(x_test_normalized)[incorrect_test][0])))
test = pd.read_csv("../input/test.csv")

test.head()
test_normalized = np.array(test).reshape(-1, 28, 28).astype(np.float32)/255
submission = pd.DataFrame({"ImageId": range(1, len(test)+1), 

                           "Label": np.argmax(model.predict(test_normalized), axis=1)})

submission.head()
submission.to_csv("submission.csv", index=False)