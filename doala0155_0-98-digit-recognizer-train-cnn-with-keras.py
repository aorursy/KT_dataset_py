import pandas as pd

import numpy as np
train_data_path = "../input/digit-recognizer/train.csv"

test_data_path = "../input/digit-recognizer/test.csv"

sample_submission_path = "../input/digit-recognizer/sample_submission.csv"
train_data = pd.read_csv(train_data_path)
train_data.head()
# get label data

train_labels = train_data.iloc[:, 0]



# get image data

train_images = train_data.iloc[:, 1:]
train_labels_onehot = pd.get_dummies(train_labels)
train_labels.head()
train_labels_onehot.head()
train_images /= 255.
train_images = train_images.values

train_labels_onehot = train_labels_onehot.values
x_train = []

for image in train_images:

    image_data = image.reshape(28, 28, 1)

    x_train.append(image_data)
x_train = np.array(x_train)

y_train = train_labels_onehot
x_train.shape
y_train.shape
from keras import models

from keras import layers
model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=(2, 2), activation="relu", input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, kernel_size=(2, 2), activation="relu"))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, kernel_size=(2, 2), activation="relu"))

model.add(layers.Flatten())

model.add(layers.Dropout(0.4))

model.add(layers.Dense(64, activation="relu"))

model.add(layers.Dropout(0.4))

model.add(layers.Dense(10, activation="softmax"))
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
model.save("digit_cnn.h5")
test_images = pd.read_csv(test_data_path)
test_images = test_images.values.astype(np.float32)
test_images /= 255.
x_test = []

for image_data in test_images:

    image = image_data.reshape(28, 28, 1)

    x_test.append(image)
x_test = np.array(x_test)
predictions = model.predict(x_test)
predictions[0]
labels = np.argmax(predictions, axis=1)
sample = pd.read_csv(sample_submission_path)
sample["Label"] = labels

submission = sample
submission.to_csv("submission.csv", index=False)