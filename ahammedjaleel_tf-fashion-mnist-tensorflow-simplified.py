import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import tensorflow as tf

from tensorflow import keras
print(tf.__version__)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

df = train.copy()

df_test = test.copy()
# Setting Random Seeds for Reproducibilty.

seed = 66

np.random.seed(seed)
from sklearn.model_selection import train_test_split
X = train.iloc[:,1:]

Y = train.iloc[:,0]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=seed)
x_train.count()
x_train.info()
x_test.info()
# The first parameter in reshape indicates the number of examples.

# We pass it as -1, which means that it is an unknown dimension and we want numpy to figure it out.



# reshape(examples, height, width, channels)

x_train = x_train.values.reshape((-1, 28, 28))

x_test = x_test.values.reshape((-1, 28, 28))



df_test.drop('label', axis=1, inplace=True)

df_test = df_test.values.reshape((-1, 28, 28, 1))


x_train.shape


class_names = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
plt.figure()

plt.imshow(x_train[1])

plt.colorbar()
x_train = x_train/255.0
x_test = x_test/255.0
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten,Dense
model = Sequential()

model.add(Flatten(input_shape = (28,28)))

model.add(Dense(128,activation = 'relu'))

model.add(Dense(10,activation = 'softmax'))
model.summary()
model.compile(optimizer = 'adam',loss ='sparse_categorical_crossentropy',metrics = ['accuracy'] )
model.fit(x_train,y_train,epochs = 10)
test_loss,test_acc = model.evaluate(x_test,y_test)
from sklearn.metrics import accuracy_score
y_pred = model.predict_classes(x_test)
accuracy_score(y_test,y_pred)
y_pred
pred = model.predict(x_test)
pred
pred[0]
np.argmax(pred[0])