import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras import Model

from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

from keras.constraints import max_norm



sns.set(style="white", context="talk")
train = pd.read_csv("../input/train.csv")



image_length = 28

image_size = image_length**2



y_train_full = train["label"].values

y_train_full = to_categorical(y_train_full, num_classes=10)

X_train_full = train.drop(labels="label", axis="columns").values

X_train_full = X_train_full.reshape((-1, image_length, image_length, 1))



test = pd.read_csv("../input/test.csv")

X_test = test.values

X_test = X_test.reshape((-1, image_length, image_length, 1))



input_shape = (image_length, image_length, 1)



X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,

                                                  test_size=.1, random_state=0)
datagen = ImageDataGenerator(rotation_range=10,

                             width_shift_range=.1,

                             height_shift_range=.1,

                             shear_range=10,

                             zoom_range=.1)

datagen.fit(X_train)
convolution_constraint = max_norm(3, axis=[0, 1, 2])

dense_constraint = max_norm(3)



model = Sequential()

model.add(Conv2D(filters=64, kernel_size=3, padding="same",

                 kernel_constraint=convolution_constraint,

                 activation="relu", input_shape=input_shape))



model.add(Conv2D(filters=64, kernel_size=3, padding="same",

                 kernel_constraint=convolution_constraint,

                 activation="relu"))

model.add(MaxPool2D())

model.add(Dropout(.5))

model.add(Conv2D(filters=128, kernel_size=3, padding="same",

                 kernel_constraint=convolution_constraint,

                 activation="relu"))



model.add(Conv2D(filters=128, kernel_size=3, padding="same",

                 kernel_constraint=convolution_constraint,

                 activation="relu"))

model.add(MaxPool2D())

model.add(Dropout(.5))

model.add(Flatten())

model.add(Dense(units=256, kernel_constraint=dense_constraint,

                activation="relu"))

model.add(Dropout(.5))

model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])
epochs = 150

batch_size = 378

history = model.fit_generator(

    datagen.flow(X_train, y_train, batch_size=batch_size),

    validation_data=(X_val, y_val),

    epochs=epochs, steps_per_epoch=X_train.shape[0] // batch_size

)
epochs = np.array(history.epoch) + 1

train_accuracies = history.history["acc"]

validation_accuracies = history.history["val_acc"]



# Add a trend line for the validation accuracies based on the reciprocal of the epoch number.

regression = LinearRegression()

epoch_features = np.array([1/epochs]).T

trend = regression.fit(epoch_features, validation_accuracies).predict(epoch_features)

asymptote = regression.intercept_



fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(epochs, train_accuracies, label="Training")

ax.axhline(asymptote, color="grey", linestyle=":")

ax.plot(epochs, trend, color="grey", linestyle="--")

ax.plot(epochs, validation_accuracies, label="Validation")



ax.legend(loc="lower right")

ax.set_ylim([.96, 1])

ax.set_xlim([0, max(epochs)])

ax.set_title("Learning curve")

ax.set_ylabel("Accuracy")

ax.set_xlabel("Epoch")

plt.show()
final_train_accuracy = np.mean(train_accuracies[-10:])

final_validation_accuracy = np.mean(validation_accuracies[-10:])

print("Final training accuracy:\t{:.4f}".format(final_train_accuracy))

print("Final validation accuracy:\t{:.4f}".format(final_validation_accuracy))
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=3, padding="same",

                 kernel_constraint=convolution_constraint,

                 activation="relu", input_shape=input_shape))



model.add(Conv2D(filters=64, kernel_size=3, padding="same",

                 kernel_constraint=convolution_constraint,

                 activation="relu"))

model.add(MaxPool2D())

model.add(Dropout(.5))

model.add(Conv2D(filters=128, kernel_size=3, padding="same",

                 kernel_constraint=convolution_constraint,

                 activation="relu"))



model.add(Conv2D(filters=128, kernel_size=3, padding="same",

                 kernel_constraint=convolution_constraint,

                 activation="relu"))

model.add(MaxPool2D())

model.add(Dropout(.5))

model.add(Flatten())

model.add(Dense(units=256, kernel_constraint=dense_constraint,

                activation="relu"))

model.add(Dropout(.5))

model.add(Dense(units=10, activation='softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])
epochs = 150

batch_size = 420

model.fit_generator(

    datagen.flow(X_train_full, y_train_full, batch_size=batch_size),

    epochs=epochs, steps_per_epoch=X_train.shape[0] // batch_size

)



predictions = model.predict(X_test).argmax(axis=1)



predictions_count = predictions.size

submission = pd.DataFrame({"Label": predictions, "ImageId": range(1, predictions_count + 1)}, columns=["ImageId", "Label"])

submission.to_csv("submission.csv", index=False)