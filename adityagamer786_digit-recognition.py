import numpy as np 

import pandas as pd 



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, optimizers, backend, models, callbacks

from keras.layers import Dense, Dropout

from keras.models import Sequential
train_raw = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_raw = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



print(f"Training size : {train_raw.shape[0]}*{train_raw.shape[1]}\n")

print(f"Test size : {test_raw.shape[0]}*{test_raw.shape[1]}")
X = train_raw.drop("label", axis=1)/255

y = train_raw["label"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

print(f"Training size : {X_train.shape[0]}*{X_train.shape[1]}\n")

print(f"Validation size : {X_valid.shape[0]}*{X_valid.shape[1]}")
model = models.Sequential()

model.add(Dense(400, input_shape=[784],activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(200, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(100, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
# plt.figure(figsize=(40,40))

plt.plot(history.history["accuracy"], label="MSE(training data)")

plt.plot(history.history["val_accuracy"], label="MSE(validation data)")

plt.legend(["train", "test"], loc="lower right")

plt.xlabel("epoch")

plt.ylabel("accuracy")

plt.show();
X_test = test_raw/255
pred = model.predict_classes(X_test)
submission = pd.DataFrame({"ImageId": np.array(range(1,28001)), "Label": pred})

submission.to_csv("/kaggle/working/submission.csv", index=False)