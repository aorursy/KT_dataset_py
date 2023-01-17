import numpy as np
import pandas as pd
import cv2
from PIL import Image

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv", index_col=None)
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv", index_col=None)
train.head()
train_labels = train.label

train_data = np.reshape(train.drop('label', axis=1).to_numpy(), (-1, 28, 28))
test_data = np.reshape(test.to_numpy(), (-1, 28, 28))
train_data.shape
import matplotlib.pyplot as plt

plt.imshow(train_data[0])
plt.show()
def prepare_data(data):
    th = 200
    # kernel = np.ones((5,5), np.uint8)
    new_data = []

    for i in data:
        img = np.float32(i)
        img[img < th] = 1
        img[img >= th] = 0
        new_data.append(img)
        # eroded = cv2.erode(thresh, kernel, iterations=5)
        # opening = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
    return new_data
train_data = prepare_data(train_data)
test_data = prepare_data(test_data)
plt.imshow(train_data[0], cmap="binary")
plt.show()
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

train_labels = lb.fit_transform(train_labels)
train_labels[0]
import tensorflow as tf
import tensorflow.keras.layers as ly
from tensorflow.keras.models import Sequential
model = Sequential([
    ly.Flatten(input_shape=(28, 28, 1)),
    ly.Dense(64, activation="relu"),
    ly.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("\kaggle/input/digit-recognizer\models\model_binary.h5",
                             monitor="val_accuracy",
                            save_best_only=True,
                            verbose=1)
history = model.fit(train_data, train_labels,
          validation_split=0.2,
          epochs=100,
          batch_size=64,
          verbose=1,
          callbacks=[checkpoint])
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["accuracy", "val_accuracy"])
plt.title("Accuracy of MNIST")
plt.show()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])
plt.title("Loss of MNIST")
plt.show()
from tensorflow.keras.models import load_model

m = load_model("\kaggle\models\model.h5")
df_sum = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
predictions = m.predict(test_data)
predictions = [np.argmax(i) for i in predictions]
df_sum["Label"] = predictions
df_sum.head()
for i in range(5):
    plt.imshow(test_data[i])
    plt.title(f"Predicted value: {df_sum.iloc[i]['Label']}")
    plt.show()
