import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import numpy as np

import pandas as pd

import math



import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

from matplotlib.pyplot import plot



style.use("seaborn-whitegrid")

%matplotlib inline

%config InlineBackend.figure_format = "retina"



from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf



from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(train.shape)

print(test.shape)
train.head(3)
test.head(3)
X = train.drop("label", axis = 1)

y = train["label"]
X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y, 

                                                    test_size = 0.25)
X_train = X_train / 255

X_test = X_test / 255

test = test / 255
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
model = Sequential()



model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu", padding = "same", input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu", padding = "same"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(128, kernel_size = (3, 3), activation = "relu", padding = "same"))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size = (3, 3), activation = "relu", padding = "same"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.3))



model.add(Conv2D(256, kernel_size = (3, 3), activation = "relu", padding = "same"))

model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size = (3, 3), activation = "relu", padding = "same"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(512, activation = "relu"))

model.add(Dense(10, activation = "softmax"))



model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = 0.0001, centered = True, momentum = 0), 

              loss = "categorical_crossentropy", 

              metrics = ["accuracy"])



history = model.fit(X_train, 

                    y_train, 

                    validation_split = 0.2,

                    epochs = 80, 

                    batch_size = 64, 

                    shuffle = True,

                    verbose = 0)



print("Test score :" + str(model.evaluate(X_test, y_test)))

print("")

print("Train score :" + str(model.evaluate(X_train, y_train)))

print(model.summary())
y_true = pd.Series(np.argmax(y_test, axis = 1), name = "actual")

y_preds = pd.Series(np.argmax(model.predict(X_test), axis = 1), name = "predictions")



pd.crosstab(y_true, y_preds)
fig, ax = plt.subplots(1, 2, figsize = (20,8))

ax[0].plot(history.history["accuracy"], label = "Train", color = "Lightblue", linewidth = 3)

ax[0].plot(history.history["val_accuracy"], label = "Test", color = "Salmon", linewidth = 3)

ax[1].plot(history.history["loss"], label = "Train", color = "Lightblue", linewidth = 3)

ax[1].plot(history.history["val_loss"], label = "Test", color = "Salmon", linewidth = 3)

ax[0].set_title("Accuracy", fontsize = 14)

ax[1].set_title("Loss", fontsize = 14)

ax[0].set_xlabel("Epoch")

ax[1].set_xlabel("Epoch")

ax[0].set_ylabel("Accuracy")

ax[1].set_ylabel("Loss")

ax[0].legend()

ax[1].legend()

plt.show()
y_sub = pd.Series(np.argmax(model.predict(test), axis = 1), name = "Label")

submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), y_sub], axis = 1)

submission.to_csv("submission.csv", index = False)

submission.head()