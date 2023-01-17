import matplotlib

matplotlib.use("Agg")

%matplotlib inline

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report

from keras.callbacks import LearningRateScheduler

from keras.optimizers import SGD

from keras.datasets import cifar10

import matplotlib.pyplot as plt

import numpy as np

import argparse

import matplotlib.pyplot as plt

from keras.layers import Conv2D, BatchNormalization, AveragePooling2D, MaxPooling2D, ZeroPadding2D

from keras.layers import Activation, Dense, Flatten, Input, add

from keras.models import Model

from keras.regularizers import l2

from keras import backend as K
class ResNet:

    @staticmethod

    def residual_module(data, K, stride, chanDim, red=False,

        reg=0.0001, bnEps=2e-5, bnMom=0.9):

        # the shortcut branch of the ResNet module should be

        # initialize as the input (identity) data

        shortcut = data



        # the first block of the ResNet module are the 1x1 CONVs

        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,

            momentum=bnMom)(data)

        act1 = Activation("relu")(bn1)

        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,

            kernel_regularizer=l2(reg))(act1)



        # the second block of the ResNet module are the 3x3 CONVs

        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,

            momentum=bnMom)(conv1)

        act2 = Activation("relu")(bn2)

        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,

            padding="same", use_bias=False,

            kernel_regularizer=l2(reg))(act2)



        # the third block of the ResNet module is another set of 1x1

        # CONVs

        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,

            momentum=bnMom)(conv2)

        act3 = Activation("relu")(bn3)

        conv3 = Conv2D(K, (1, 1), use_bias=False,

            kernel_regularizer=l2(reg))(act3)



        # if we are to reduce the spatial size, apply a CONV layer to

        # the shortcut

        if red:

            shortcut = Conv2D(K, (1, 1), strides=stride,

                use_bias=False, kernel_regularizer=l2(reg))(act1)



        # add together the shortcut and the final CONV

        x = add([conv3, shortcut])



        # return the addition as the output of the ResNet module

        return x



    @staticmethod

    def build(width, height, depth, classes, stages, filters,

        reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):

        # initialize the input shape to be "channels last" and the

        # channels dimension itself

        inputShape = (height, width, depth)

        chanDim = -1



        # if we are using "channels first", update the input shape

        # and channels dimension

        if K.image_data_format() == "channels_first":

            inputShape = (depth, height, width)

            chanDim = 1



        # set the input and apply BN

        inputs = Input(shape=inputShape)

        x = BatchNormalization(axis=chanDim, epsilon=bnEps,

            momentum=bnMom)(inputs)



        # apply a single CONV layer

        x = Conv2D(filters[0], (3, 3), use_bias=False,

            padding="same", kernel_regularizer=l2(reg))(x)



        # loop over the number of stages

        for i in range(0, len(stages)):

            # initialize the stride, then apply a residual module

            # used to reduce the spatial size of the input volume

            stride = (1, 1) if i == 0 else (2, 2)

            x = ResNet.residual_module(x, filters[i + 1], stride,

                chanDim, red=True, bnEps=bnEps, bnMom=bnMom)



            # loop over the number of layers in the stage

            for j in range(0, stages[i] - 1):

                # apply a ResNet module

                x = ResNet.residual_module(x, filters[i + 1],

                    (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)



        # apply BN => ACT => POOL

        x = BatchNormalization(axis=chanDim, epsilon=bnEps,

            momentum=bnMom)(x)

        x = Activation("relu")(x)

        x = AveragePooling2D((8, 8))(x)



        # softmax classifier

        x = Flatten()(x)

        x = Dense(classes, kernel_regularizer=l2(reg))(x)

        x = Activation("softmax")(x)



        # create the model

        model = Model(inputs, x, name="resnet")



        # return the constructed network architecture

        return model
#Base Class having a simple plot function

class LearningRateDecay():

    def plot(self, epochs, title="LR Schedule"):

        rates = [self(i) for i in epochs]

        plt.style.use("ggplot")

        plt.figure()

        plt.plot(epochs, rates)

        plt.xlabel("Epochs")

        plt.ylabel("Rates")

        plt.title(title)
class StepDecay(LearningRateDecay):

    def __init__(self, initial_alpha=0.01, factor=0.25, drop_every=10):

        self.initial_alpha = initial_alpha

        self.factor = factor

        self.drop_every = drop_every



    def __call__(self, epoch):

        exp = np.floor((1 + epoch) / self.drop_every)

        new_alpha = self.initial_alpha * (self.factor ** exp)



        return float(new_alpha)
class PolynomialDecay(LearningRateDecay):

    def __init__(self, max_epochs=100, initial_alpha=0.01, power=1.0):

        self.max_epochs = max_epochs

        self.initial_alpha = initial_alpha

        self.power = power



    def __call__(self, epoch):

        decay = (1 - (epoch / float(self.max_epochs))) ** self.power

        new_alpha = self.initial_alpha * decay



        return float(new_alpha)
schedule_type = "linear" #step, linear, poly

epochs = 50

lr = 0.1

path_to_save_plot = "."
schedule = None

if schedule_type=="step":

    schedule = StepDecay(initial_alpha=1e-1, factor=0.25, drop_every=15)

elif schedule_type=="linear":

    schedule = PolynomialDecay(max_epochs=epochs, initial_alpha=1e-1, power=1)

else:

    schedule = PolynomialDecay(max_epochs=epochs, initial_alpha=1e-1, power=5)
callbacks = [LearningRateScheduler(schedule)]
((train_X, train_y), (test_X, test_y)) = cifar10.load_data()
train_X = train_X.astype("float")/255.

test_X = test_X.astype("float")/255.
lb = LabelBinarizer()

train_y = lb.fit_transform(train_y)

test_y = lb.transform(test_y)
#The order of occurence of these classes is very important

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
opt = SGD(lr=1e-1, momentum=0.9)
model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),

    batch_size=128, epochs=epochs, callbacks=callbacks, verbose=1)
N = np.arange(0, epochs)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, H.history["loss"], label="training_loss")

plt.plot(N, H.history["val_loss"], label="val_loss")

plt.plot(N, H.history["acc"], label="training_acc")

plt.plot(N, H.history["val_acc"], label="val_acc")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

# plt.show()
plt.savefig(".")
if schedule is not None:

    schedule.plot(N)