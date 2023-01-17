import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation

from tensorflow.keras.optimizers import SGD

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/final.csv", header=None)
X = df[df.columns[0:10]].values

y = df[df.columns[10:]].values

print(X.shape,y.shape)
max(y)
y


Y = (tf.keras.utils.to_categorical(y, num_classes=9))

def one_hot_encode(labels):

    n_labels = len(labels)

    n_unique_labels = len(np.unique(labels))

    one_hot_encode = np.zeros((n_labels,n_unique_labels))

    one_hot_encode[np.arange(n_labels),labels] = 1

    return one_hot_encode
def read_dataset():

    dir_path = ""

    df = pd.read_csv("../input/final.csv", header=None)



    X = df[df.columns[0:10]].values

    y = df[df.columns[10:]].values



    encoder = LabelEncoder()

    encoder.fit(y)



    y = encoder.transform(y)

    Y = one_hot_encode(y)



    return (X,Y)
# X, Y = read_dataset()
print(type(Y))

print(Y.shape)

print(Y.size)

print(len(Y))

Y

X[5:6]
Y[5]
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
# train_x = np.random.random((100, 10))

# test_x = np.random.random((100, 10))

# train_y = tf.keras.utils.to_categorical(np.random.randint(9, size=(100, 1)), num_classes=9)

# test_y = tf.keras.utils.to_categorical(np.random.randint(9, size=(100, 1)), num_classes=9)
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=10))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(9, activation='softmax'))



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
train_x[5:6]
train_y[5]
# model.compile(loss='categorical_crossentropy', 

#               optimizer=sgd,

#               matrices=['accuracy'])
model.compile(loss='categorical_crossentropy', 

              optimizer='adam',

              matrices=['accuracy'])
model.fit(train_x, train_y, epochs=1000, batch_size=128)
score = model.evaluate(test_x, test_y, batch_size=128)
model.metrics_names
score
# print(train_y[5:6])

model.predict(train_x[5:6], 

                  batch_size=None, 

                  verbose=0, 

                  steps=None)
for i in range(10):

    plt.plot(test_x[i]) # plotting by columns

    plt.title(test_x[i])

    plt.show()

    p = model.predict(train_x[i:i+1], 

                  batch_size=None, 

                  verbose=0, 

                  steps=None)

    print(p)

    k = (tf.argmax(p, 1))

    print(k)

    print(test_y[i])


