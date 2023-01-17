from keras.models import Sequential

from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

import numpy as np

import pandas as pd

np.random.seed(7)

df = pd.read_csv("../input/iris.csv")

X = df.drop(["class"],axis=1)

Y = df['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)



dummy_y = np_utils.to_categorical(encoded_Y)
model = Sequential()

model.add(Dense(8, input_dim=4, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, dummy_y, epochs=150, batch_size=5)
scores = model.evaluate(X, dummy_y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(X)
# 0 = setosa , 1 = versicolor , 2 = virginica

test = np.argmax(predictions,axis=1)

test