!pip install nb_black -q
%load_ext nb_black
import pandas as pd

import numpy as np



sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")



test_backup = test





def verify_balck_white(pixel):

    return 0 if pixel < 127 else 1
import matplotlib.pyplot as plt



y_train = train[["label"]]

x_train = train.drop(["label"], axis=1)

x_train = x_train.applymap(verify_balck_white)

test = test.applymap(verify_balck_white)



plt.figure(figsize=(25, 15))

for img in range(40):

    plt.subplot(4, 10, img + 1)

    a = list(x_train.loc[img])

    ax = np.reshape(a, (28, 28), order="C")

    plt.imshow(ax)

    

    

#Reshape

x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

import keras

from keras.models import Sequential

from keras import optimizers

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator





model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))

model.add(Conv2D(128, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation="softmax"))



stoper = [keras.callbacks.EarlyStopping(monitor="loss")]

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(

    optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["mse", "accuracy"],

)



# Trainig and returning back the results.

history = model.fit(

    x_train,

    y_train,

    epochs=30,

    verbose=1,

    validation_split=0.05,

    shuffle=True,

    callbacks=stoper,

    use_multiprocessing=True,

)
model.summary()
import plotly.express as px

import matplotlib.pyplot as plt





def plot_train_statics(history, tags=None):

    dfs = []

    tags = tags if tags != None else list(history.history)



    for tag in tags:

        df = pd.DataFrame(history.history[tag], columns=["value"])

        df["epoch"] = df.index

        df["color"] = tag

        dfs.append(df)



    df = pd.concat(dfs)

    df.value = df.value.round(2)

    fig = px.line(df, x="epoch", y="value", color="color", title="Training Curve")

    fig.show()
plot_train_statics(history, ["loss", "accuracy"])

plot_train_statics(history, ["mse"])
prediction = model.predict_classes(test)

sample_submission["Label"] = prediction

sample_submission.to_csv("submission_digit.csv", index=False)

sample_submission.head(20)
plt.figure(figsize=(25, 15))

for img in range(20):

    plt.subplot(4, 10, img + 1)

    a = list(test_backup.loc[img])

    ax = np.reshape(a, (28, 28), order="C")

    plt.imshow(ax)