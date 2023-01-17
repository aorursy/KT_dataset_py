import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
xt = test_data.to_numpy().reshape((28000, 28, 28, 1))

# ----

xs = train_data.drop("label", axis=1).to_numpy().reshape((42000, 28, 28, 1))

ys = train_data["label"].to_numpy().reshape([42000, 1])

print(xt.shape, xs.shape, ys.shape)
ys_oh = keras.utils.to_categorical(ys)
ys_oh[0]
#You can choose other values for "m"

m=99



figure, ax = plt.subplots(1,2)



ax[0].imshow(xs[m])

ax[0].set_title(f"Label = {ys[m]}")

ax[1].imshow(xt[m])

ax[1].set_title(r"Test data")
model = keras.models.Sequential([keras.layers.Conv2D(4, (3,3), padding="valid", activation="relu", input_shape=(28,28,1)),

                                 keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid"),

                                 keras.layers.Conv2D(8, (3,3), padding="valid", activation="relu"),

                                 keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid"),

                                 keras.layers.Flatten(),

                                 keras.layers.Dense(32, activation=tf.nn.tanh),

                                 keras.layers.Dense(10, activation=tf.nn.softmax)])



model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit(xs, ys_oh, epochs=10)
raw_preds = model.predict(xt)

preds = np.argmax(raw_preds, axis=1)
# Check prediction manually

m = 4949



plt.imshow(xt[m].reshape(28,28))

plt.title(f"Prediction label = {preds[m]}")
sub = np.concatenate([np.arange(1,preds.size+1).reshape([28000, 1]), preds.reshape([preds.size, 1])], axis=1)

final_preds = pd.DataFrame(data=sub, columns = ["ImageId", "Label"])

final_preds.to_csv("submission1.csv", index=False)