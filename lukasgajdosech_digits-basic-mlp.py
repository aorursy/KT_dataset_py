import numpy as np

import pandas as pd 

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.utils import to_categorical
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)

X_train = X_train / 255.0

test = test / 255.0

print(f"{X_train.shape} {Y_train.shape}")
mlp = Sequential()

mlp.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))

mlp.add(Dense(100, activation='relu'))

mlp.add(Dense(10, activation='softmax'))

mlp.compile(loss='categorical_crossentropy', optimizer=Adam(0.004), metrics=['accuracy'])

mlp.fit(X_train, to_categorical(Y_train), epochs=10, validation_split=0.1, verbose=True)
prediction = mlp.predict(test)

prediction = np.argmax(prediction, axis = 1)

prediction = pd.Series(prediction, name="Label")

prediction = pd.concat([pd.Series(range(1,28001), name = "ImageId"), prediction], axis = 1)

prediction.to_csv("mnist_mlp.csv", index=False); print(prediction[:10])