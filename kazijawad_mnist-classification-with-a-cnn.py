from keras.datasets import mnist



# Load MNIST dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Initial shape of training data:", str(x_train.shape))

print()

print("Number of samples in our training data:", str(len(x_train)))

print("Number of labels in our training data:", str(len(y_train)))

print()

print("Number of samples in our test data:", str(len(x_test)))

print("Number of labels in our test data:", str(len(y_test)))

print()

print("Dimensions of our training data:", str(x_train[0].shape))

print("Labels in our training data:", str(y_train.shape[0]))

print()

print("Dimensions of our test data:", str(x_test[0].shape))

print("Labels in our test data:", str(y_test.shape[0]))
import numpy as np

import matplotlib.pyplot as plt



for index in range(1, 7):

    random_int = np.random.randint(0, len(x_train))

    plt.subplot(330 + index)

    plt.imshow(x_train[random_int], cmap=plt.get_cmap("gray"))



plt.show()
# Store the number of rows and columns

img_rows = x_train[0].shape[0]

img_cols = x_train[1].shape[0]



# A 4th dimension is necessary for Keras input

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)



# Store the shape of a single image

input_shape = (img_rows, img_cols, 1)



# Change image type to the float32 data type

x_train = x_train.astype("float32")

x_test = x_test.astype("float32")



# Normalize the image by converting the range from (0, 256) to (0, 1)

x_train /= 255

x_test /= 255



print("Training data shape:", x_train.shape)

print(x_train.shape[0], "training samples")

print(x_test.shape[0], "test samples")
from keras.utils import np_utils



y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)



num_classes = y_test.shape[1]

num_pixels = x_train.shape[1] * x_train.shape[2]



print("Number of classes:", str(num_classes))
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras import backend as K

from keras.optimizers import SGD



model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))

model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation="softmax"))



model.compile(loss="categorical_crossentropy", optimizer=SGD(0.01), metrics=["accuracy"])



model.summary()
batch_size = 32

epochs = 10



history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print("Test loss:", score[0])

print("Test accuracy:", score[1])
# Plot Loss Chart

history_dict = history.history



loss_values = history_dict["loss"]

val_loss_values = history_dict["val_loss"]

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label="Validation/Test Loss")

line2 = plt.plot(epochs, loss_values, label="Training Loss")

plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)

plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.grid(True)

plt.legend()

plt.show
# Plot Accuracy Chart

acc_values = history_dict["accuracy"]

val_acc_values = history_dict["val_accuracy"]



line1 = plt.plot(epochs, val_acc_values, label="Validation/Test Accuracy")

line2 = plt.plot(epochs, acc_values, label="Training Accuracy")

plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)

plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.grid(True)

plt.legend()

plt.show
model.save("model.h5")

print("Model Saved")
from keras.models import load_model



classifier = load_model("model.h5")

print("Model Loaded")
from keras.utils.vis_utils import plot_model



# Generate plot

plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)



img = plt.imread("model.png")

plt.figure(figsize=(30,15))

imgplot = plt.imshow(img)