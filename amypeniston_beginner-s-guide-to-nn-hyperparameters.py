import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



SEED = 42
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

y = train.pop("label")
train.shape, test.shape
train.head()
X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.2, random_state=SEED)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
X_train.head()
for i in range(12):

    plt.subplot(3, 4, i+1)

    image = X_train.iloc[i].values.reshape((28, 28))

    plt.imshow(image, cmap="Greys")

    plt.axis("off")

plt.tight_layout()

plt.show()
list(y_train[0:12])
X_train = X_train / 255

X_valid = X_valid / 255
n = 10

y_train = keras.utils.to_categorical(y_train, num_classes=n)

y_valid = keras.utils.to_categorical(y_valid, num_classes=n)
y_train.shape, y_valid.shape
model = Sequential()

model.add(Dense(64, activation="sigmoid", input_shape=(784,)))

model.add(Dense(10, activation="softmax"))
model.summary()
784 * 64 + 64 # 1 parameter for each of the 784 input neurons * 64 hidden neurons + 64 additional bias terms
64 * 10 + 10 # 1 parameter for each of the 64 hidden layer neurons * 10 output neurons + 10 additional bias terms
(784 * 64 + 64) + (64 * 10 + 10) # total number of parameters
model.compile(loss="mean_squared_error", optimizer=SGD(lr=0.01), metrics=["accuracy"])
hist = model.fit(x=X_train,

          y=y_train,

          batch_size=128,

          epochs=50, # Adjust to see whether validation metrics continue to improve or start to overfit

          verbose=0, # Adjust to see training progress

          validation_data=(X_valid, y_valid))
val_loss = hist.history["val_loss"]

val_accuracy = hist.history["val_accuracy"]



fig, ax = plt.subplots(1, 2, figsize=(15, 5))



sns.lineplot(x=range(0,len(val_accuracy)), y=val_accuracy, ax=ax[0], label="Validation Accuracy")

sns.lineplot(x=range(0,len(val_loss)), y=val_loss, ax=ax[1], label="Validation Loss")



ax[0].set_xlabel("# of Epochs")

ax[1].set_xlabel("# of Epochs")



plt.suptitle("Learning Curves")

plt.show()
model.evaluate(X_valid, y_valid)
# model.save("model.hd5")

# model = keras.models.load_model("model.hd5")

# model.summary()
class NeuralNetwork():

    def __init__(self, name, batch_size, epochs, learning_rate, verbose):

        self.name = name

        self.batch_size = batch_size

        self.epochs = epochs

        self.learning_rate = learning_rate

        self.verbose = verbose

        self.model = Sequential()

        

    def add_(self, layer):

        self.model.add(layer)



    def compile_and_fit(self):

        self.model.compile(loss="mean_squared_error", optimizer=SGD(lr=self.learning_rate), metrics=["accuracy"])

        self.history = self.model.fit(x=X_train,

                                      y=y_train,

                                      batch_size=self.batch_size,

                                      epochs=self.epochs,

                                      verbose=self.verbose,

                                      validation_data=(X_valid, y_valid))

        self.val_loss = self.history.history["val_loss"]

        self.val_accuracy = self.history.history["val_accuracy"]

    

    def plot_learning_curves(self):

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        

        sns.lineplot(x=range(0,len(self.val_accuracy)), y=self.val_accuracy, ax=ax[0], label="Validation Accuracy")

        sns.lineplot(x=range(0,len(self.val_loss)), y=self.val_loss, ax=ax[1], label="Validation Loss")



        ax[0].set_xlabel("# of Epochs")

        ax[1].set_xlabel("# of Epochs")



        plt.suptitle("Learning Curves: {}".format(self.name))

        plt.show()



    def evaluate_(self):

        return self.model.evaluate(X_valid, y_valid)

    

    def save(self, filename):

        self.model.save("working/"+filename+".hd5")

        

    def summary_(self):

        return self.model.summary()
def compare_learning_curves(models):

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    

    for model in models:

        sns.lineplot(x=range(0,len(model.val_accuracy)), y=model.val_accuracy, ax=ax[0], label=model.name)

        sns.lineplot(x=range(0,len(model.val_loss)), y=model.val_loss, ax=ax[1], label=model.name)

    

    ax[0].set_xlabel("# of Epochs")

    ax[1].set_xlabel("# of Epochs")



    ax[0].set_title("Validation Accuracy")

    ax[1].set_title("Validation Loss")



    plt.suptitle("Learning Curves")

    plt.show()
batch_sizes = [8, 16, 32, 64, 128, 256]

n_epochs = 50
accuracy = pd.DataFrame(columns=batch_sizes, index=range(n_epochs))

loss = pd.DataFrame(columns=batch_sizes, index=range(n_epochs))

accuracy["Epoch"] = range(n_epochs)

loss["Epoch"] = range(n_epochs)
for batch_size in batch_sizes:

    model = Sequential()

    model.add(Dense(64, activation="sigmoid", input_shape=(784,)))

    model.add(Dense(10, activation="softmax"))

    model.compile(loss="mean_squared_error", optimizer=SGD(lr=0.01), metrics=["accuracy"])

    

    hist = model.fit(x=X_train,

              y=y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              verbose=0,

              validation_data=(X_valid, y_valid))

    

    accuracy[batch_size] = hist.history["val_accuracy"]

    loss[batch_size] = hist.history["val_loss"]
accuracy_melt = accuracy.melt(value_name="Accuracy", var_name="Batch Size", id_vars=["Epoch"])

loss_melt = loss.melt(value_name="Loss", var_name="Batch Size", id_vars=["Epoch"])



accuracy_melt["Batch Size"] = accuracy_melt["Batch Size"].astype(object)

loss_melt["Batch Size"] = loss_melt["Batch Size"].astype(object)
accuracy_melt = accuracy.melt(value_name="Accuracy", var_name="Batch Size", id_vars=["Epoch"])

loss_melt = loss.melt(value_name="Loss", var_name="Batch Size", id_vars=["Epoch"])
fig, ax = plt.subplots(1, 2, figsize=(20, 8))



sns.lineplot(x="Epoch", y="Accuracy", hue="Batch Size", data=accuracy_melt, ax=ax[0], legend="full")

sns.lineplot(x="Epoch", y="Loss", hue="Batch Size", data=loss_melt, ax=ax[1], legend="full")



ax[0].set_title("Validation Accuracy")

ax[1].set_title("Validation Loss")



ax[0].set_xlabel("# of Epochs")

ax[1].set_xlabel("# of Epochs")



plt.suptitle("Learning Curves")

plt.show()
n_epochs = 100

batch_size = 128

verbose = 0
learning_rate = 100

nn_lr_100 = NeuralNetwork("LR = {}".format(learning_rate), batch_size, n_epochs, learning_rate, verbose)

nn_lr_100.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_lr_100.add_(Dense(10, activation="softmax"))



learning_rate = 1000

nn_lr_1000 = NeuralNetwork("LR = {}".format(learning_rate), batch_size, n_epochs, learning_rate, verbose)

nn_lr_1000.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_lr_1000.add_(Dense(10, activation="softmax"))



learning_rate = 10

nn_lr_10 = NeuralNetwork("LR = {}".format(learning_rate), batch_size, n_epochs, learning_rate, verbose)

nn_lr_10.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_lr_10.add_(Dense(10, activation="softmax"))



learning_rate = 1

nn_lr_1 = NeuralNetwork("LR = {}".format(learning_rate), batch_size, n_epochs, learning_rate, verbose)

nn_lr_1.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_lr_1.add_(Dense(10, activation="softmax"))



learning_rate = 0.1

nn_lr_p1 = NeuralNetwork("LR = {}".format(learning_rate), batch_size, n_epochs, learning_rate, verbose)

nn_lr_p1.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_lr_p1.add_(Dense(10, activation="softmax"))



learning_rate = 0.01 # default

nn_lr_p01 = NeuralNetwork("LR = {}".format(learning_rate), batch_size, n_epochs, learning_rate, verbose)

nn_lr_p01.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_lr_p01.add_(Dense(10, activation="softmax"))



learning_rate = 0.001

nn_lr_p001 = NeuralNetwork("LR = {}".format(learning_rate), batch_size, n_epochs, learning_rate, verbose)

nn_lr_p001.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_lr_p001.add_(Dense(10, activation="softmax"))



nn_lr_100.compile_and_fit()

nn_lr_1000.compile_and_fit()

nn_lr_10.compile_and_fit()

nn_lr_1.compile_and_fit()

nn_lr_p1.compile_and_fit()

nn_lr_p01.compile_and_fit()

nn_lr_p001.compile_and_fit()
learning_rate = 0.1

nn_lr_p1 = NeuralNetwork("LR = {}".format(learning_rate), batch_size, n_epochs, learning_rate, verbose)

nn_lr_p1.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_lr_p1.add_(Dense(10, activation="softmax"))

nn_lr_p1.compile_and_fit()
compare_learning_curves([nn_lr_1000, nn_lr_100, nn_lr_10, nn_lr_1, nn_lr_p1, nn_lr_p01, nn_lr_p001])
nn_lr_10.plot_learning_curves()
nn_lr_p1.plot_learning_curves()
n_epochs = 100

batch_size = 128

learning_rate = 0.01

verbose = 0
nn_l1 = NeuralNetwork("1 Hidden Layer", batch_size, n_epochs, learning_rate, verbose)

nn_l1.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_l1.add_(Dense(10, activation="softmax"))



nn_l1.summary_()

nn_l1.compile_and_fit()

nn_l1.plot_learning_curves()
nn_l2 = NeuralNetwork("2 Hidden Layers", batch_size, n_epochs, learning_rate, verbose)

nn_l2.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_l2.add_(Dense(64, activation="sigmoid"))

nn_l2.add_(Dense(10, activation="softmax"))



nn_l2.summary_()

nn_l2.compile_and_fit()

nn_l2.plot_learning_curves()
nn_l3 = NeuralNetwork("3 Hidden Layers", batch_size, n_epochs, learning_rate, verbose)

nn_l3.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_l3.add_(Dense(64, activation="sigmoid"))

nn_l3.add_(Dense(64, activation="sigmoid"))

nn_l3.add_(Dense(10, activation="softmax"))



nn_l3.summary_()

nn_l3.compile_and_fit()

nn_l3.plot_learning_curves()
compare_learning_curves([nn_l1, nn_l2, nn_l3])
n_epochs = 100

batch_size = 128

learning_rate = 0.01

verbose = 0
nn_sigmoid = NeuralNetwork("Sigmoid", batch_size, n_epochs, learning_rate, verbose)

nn_sigmoid.add_(Dense(64, activation="sigmoid", input_shape=(784,)))

nn_sigmoid.add_(Dense(10, activation="softmax"))

nn_sigmoid.compile_and_fit()



nn_tanh = NeuralNetwork("Tanh", batch_size, n_epochs, learning_rate, verbose)

nn_tanh.add_(Dense(64, activation="tanh", input_shape=(784,)))

nn_tanh.add_(Dense(10, activation="softmax"))

nn_tanh.compile_and_fit()



nn_relu = NeuralNetwork("ReLU", batch_size, n_epochs, learning_rate, verbose)

nn_relu.add_(Dense(64, activation="relu", input_shape=(784,)))

nn_relu.add_(Dense(10, activation="softmax"))

nn_relu.compile_and_fit()
compare_learning_curves([nn_sigmoid, nn_tanh, nn_relu])