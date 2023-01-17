import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import keras

import os

import cv2

from PIL import Image

from IPython.display import FileLink
print(os.listdir("../input"))
dataset = pd.read_csv("../input/train.csv")

dataset.head()
grouped_dataset = dataset.groupby("has_cactus")

grouped_dataset.count()
dataset.count()
# Put dataset inside X and Y

def datagen(dataset=dataset, path="../input/train/train/"):

    x = np.ones((17500, 224, 224, 3), dtype=np.uint8)

    y = np.ones(17500)

    

    counter = 0

    for rec in dataset.values:

        img = cv2.imread(path + rec[0])

        img = cv2.resize(img, (32, 32))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.copyMakeBorder(img, 96, 96, 96, 96, cv2.BORDER_REFLECT)

        x[counter] = img

        y[counter] = rec[1]



        counter += 1

    

    permutation = np.random.permutation(x.shape[0])

    x = x[permutation]

    y = y[permutation]

    

    return x, y
X, Y = datagen()

X.shape, Y.shape
# Print images containing cactus

fig, axs = plt.subplots(1, 5, figsize=(25, 5))



for ax, img, label in zip(axs, X[10:15], Y[10:15]):

    label = "cactus" if label == 1.0 else "No cactus"

    ax.set_title(label)

    ax.imshow(img)

    # ax.grid(True)



plt.show()
# Y = keras.utils.to_categorical(Y)

# X.shape, Y.shape
# model = keras.models.Sequential()

# model.add(keras.layers.Conv2D(10, 5, padding="valid", input_shape=(32, 32, 3)))

# model.add(keras.layers.MaxPooling2D(2, 2))

# model.add(keras.layers.ReLU())

# model.add(keras.layers.Conv2D(20, 5, padding="valid"))

# model.add(keras.layers.SpatialDropout2D(0.5))

# model.add(keras.layers.MaxPooling2D(2, 2))

# model.add(keras.layers.ReLU())



# model.add(keras.layers.Flatten())

# model.add(keras.layers.Dense(320, activation="relu"))

# model.add(keras.layers.Dropout(0.5))

# model.add(keras.layers.Dense(1, activation="sigmoid"))



# model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.00001), metrics=["accuracy"])
# model.fit(X, Y, batch_size=256, epochs=50, verbose=1, validation_split=0.2)
test_imgs = os.listdir("../input/test/test/")

print(len(test_imgs), test_imgs[:5])
densenet = keras.applications.densenet.DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
base_model = densenet.layers[-2].output

prediction = keras.layers.Dense(1, activation="sigmoid")(base_model)



densenet_model = keras.models.Model(inputs=densenet.input, outputs=prediction)

densenet_model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.00001), metrics=["accuracy"])

densenet_model.summary()
densenet_model.fit(X, Y, batch_size=16, epochs=5, verbose=1, validation_split=0.2)
densenet_model.fit(X, Y, batch_size=16, epochs=5, verbose=1, validation_split=0.2)
# Put test inside z

def test_pred(test_imgs=test_imgs, path="../input/test/test/"):

    results = []

    

    for rec in test_imgs:

        img = cv2.imread(path + rec)

        img = cv2.resize(img, (32, 32))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.copyMakeBorder(img, 96, 96, 96, 96, cv2.BORDER_REFLECT)

        img = np.reshape(img, (1, 224, 224, 3))

        result = densenet_model.predict(img, batch_size=1)

        # result = 1 if result[0][0] >= 0.5 else 0

        # results.append([rec, np.clip(result, 0.0005, 0.9995)])

        results.append([rec, result[0][0]])

    

    return results
predictions = test_pred()

predictions = pd.DataFrame(predictions, columns=["id", "has_cactus"])

predictions.head()
# predictions = test_pred()

# predictions = pd.DataFrame(predictions, columns=["id", "has_cactus"])

# predictions.head()
predictions.to_csv("densetmodel_submissions.csv", index=False)