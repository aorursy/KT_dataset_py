# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import tensorflow as tf



from tensorflow.keras.utils import to_categorical

from tensorflow.keras import models, layers



from sklearn.model_selection import train_test_split
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")



train_label = train_df.pop("label")

train_df = train_df.values



train_dfX, test_dfX, train_label, test_label = train_test_split(

    train_df, train_label, test_size=0.2, random_state=3)



train_dfX = train_dfX.reshape(33600, 28, 28)

train_dfX.shape
test_df.head()
test_df = test_df.values

test_df = test_df.reshape((test_df.shape[0], 28, 28, 1))

test_df.shape
for i in range(9):

    plt.subplot(330 + 1 + i)

    plt.imshow(train_dfX[i], cmap=plt.get_cmap("gray"))

    

plt.show()
train_dfX = train_dfX.reshape((train_dfX.shape[0], 28, 28, 1))

test_dfX = test_dfX.reshape((test_dfX.shape[0], 28, 28, 1))

train_label = to_categorical(train_label)

test_label = to_categorical(test_label)
train_dfX = train_dfX.astype("float32")

test_dfX = test_dfX.astype("float32")

train_dfX = train_dfX / 255.0

test_dfX = test_dfX / 255.0
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform",

                            input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_uniform"),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(10, activation="softmax")

])



optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)



model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(),

             metrics=["accuracy"])



history = model.fit(train_dfX, train_label, epochs=7, batch_size=32,

                   validation_data=(test_dfX, test_label))
hist = pd.DataFrame(history.history)

hist["epoch"] = history.epoch





def plot_history(history):

    plt.figure()

    plt.xlabel("Epochs")

    plt.ylabel("Train, Val Accuracy")

    plt.plot(hist["epoch"], hist["accuracy"], label="Train Acc")

    plt.plot(hist["epoch"], hist["val_accuracy"], label="Val Acc")

    plt.ylim([0, 1])

    plt.legend()

    

    plt.figure()

    plt.xlabel("Epochs")

    plt.ylabel("Train, Val Loss")

    plt.plot(hist["epoch"], hist["loss"], label="Train Loss")

    plt.plot(hist["epoch"], hist["val_loss"], label="Val Loss")

    plt.ylim([0, 0.3])

    plt.legend()

    plt.show()







plot_history(hist)    
eval_hist = model.evaluate(test_dfX, test_label, verbose=2)

plot_history(eval_hist)
pred = model.predict(test_df)



pred = [np.argmax(y, axis=None, out=None) for y in pred]



pred = pd.DataFrame(pred)
sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

sub["Label"] = pred

sub.to_csv("my_submission.csv", index=False)