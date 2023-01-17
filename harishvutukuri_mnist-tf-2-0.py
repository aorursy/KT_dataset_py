import gc

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, Activation, Dense, BatchNormalization, concatenate, AveragePooling2D, Dropout, Input, Flatten, MaxPooling2D



import warnings

warnings.filterwarnings('ignore')
# SEQUENTIAL API

def mnist_sequential(width, height, depth, classes):

    # initialize the model along with the input shape

    model = Sequential()

    inputShape = (height, width, depth)

    

    # define the first (and only) CONV => RELU layer

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=inputShape))

    model.add(Activation("relu"))

    

    # softmax classifier

    model.add(Flatten())

    model.add(Dense(classes))

    model.add(Activation("softmax"))

    

    return model
# FUNCTIONAL API

def mnist_functional(width, height, depth, classes):

    

    def conv_module(x, f, kX, kY, stride, changeDim, padding="same"):

        # define a CONV => BN => RELU pattern

        x = Conv2D(filters=f, kernel_size=(kX, kY), strides=stride, padding=padding) (x)

        x = BatchNormalization(axis=changeDim) (x)

        x = Activation("relu") (x)

        

        return x

    

    def inception_module(x, numK1x1, numK3x3, changeDim):

        # define two CONV modules, then concatenate across the channel dimension

        

        conv_1x1 = conv_module(x, numK1x1, 1, 1, (1, 1), changeDim)

        conv_3x3 = conv_module(x, numK3x3, 3, 3, (1, 1), changeDim)

        x = concatenate([conv_1x1, conv_3x3], axis=changeDim)

        

        return x

    

    def downsample_module(x, f, changeDim):

        # define the CONV module and POOL, then concatenate across the channel dimensions

        

        conv_3x3 = conv_module(x, f, 3, 3, (2, 2), changeDim, padding="valid")

        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = concatenate([conv_3x3, pool], axis=changeDim)

        

        return x

    

    inputShape = (height, width, depth)

    changeDim = -1

    

    # define the model input and first CONV module

    inputs = Input(shape=inputShape)

    x = conv_module(inputs, 96, 3, 3, (1, 1), changeDim)

    

    # two Inception modules followed by a downsample module

    x = inception_module(x, 32, 32, changeDim)

    x = inception_module(x, 32, 48, changeDim)

    x = downsample_module(x, 80, changeDim)

    

    # four Inception modules followed by a downsample module

    x = inception_module(x, 112, 48, changeDim)

    x = inception_module(x, 96, 64, changeDim)

    x = inception_module(x, 80, 80, changeDim)

    x = inception_module(x, 48, 96, changeDim)

    x = downsample_module(x, 96, changeDim)

    

    # two Inception modules followed by global POOL and dropout

    x = inception_module(x, 176, 160, changeDim)

    x = inception_module(x, 176, 160, changeDim)

    x = AveragePooling2D((7, 7))(x)

    x = Dropout(0.5)(x)

    

    # softmax classifier

    x = Flatten()(x)

    x = Dense(classes)(x)

    x = Activation("softmax")(x)

    

    # create model

    model = Model(inputs, x, name='minigooglenet')

    

    return model
# MODEL SUBCLASSING

class MnistModel(Model):

    def __init__(self, classes):

        super(MnistModel, self).__init__()

        

        # initialize the layers in the first (CONV => RELU) * 2 => POOL + DROP 

        self.conv1A = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same")

        self.act1A = Activation("relu")

        self.conv1B = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding="same")

        self.act1B = Activation("relu")

        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        self.do1 = Dropout(0.25)

        

        # initialize the layers in the second (CONV => RELU) * 2 => POOL + DROP 

        self.conv2A = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")

        self.act2A = Activation("relu")

        self.conv2B = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same")

        self.act2B = Activation("relu")

        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        self.do2 = Dropout(0.25)

        

        # initialize the layers in our fully-connected layer set

        self.flatten = Flatten()

        self.dense3 = Dense(256)

        self.act3 = Activation("relu")

        self.do3 = Dropout(0.25)

        

        # initialize the layers in the softmax classifier layer set

        self.dense4 = Dense(classes)

        self.softmax = Activation("softmax")

        

    def call(self, inputs):

        # build the first (CONV => RELU) * 2 => POOL + DROP layer set

        x = self.conv1A(inputs)

        x = self.act1A(x)

        x = self.conv1B(x)

        x = self.act1B(x)

        x = self.pool1(x)

        x = self.do1(x)

        

        # build the second (CONV => RELU) * 2 => POOL + DROP layer set

        x = self.conv2A(x)

        x = self.act2A(x)

        x = self.conv2B(x)

        x = self.act2B(x)

        x = self.pool2(x)

        x = self.do2(x)

        

        # build our FC layer set

        x = self.flatten(x)

        x = self.dense3(x)

        x = self.act3(x)

        x = self.do3(x)

        

        # build the softmax classifier

        x = self.dense4(x)

        x = self.softmax(x)

        

        return x
# initialize the initial learning rate, batch size, and number of epochs to train

INIT_LR = 1e-2

BATCH_SIZE = 100

NUM_EPOCHS = 30



# Reading Train and Test data

df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')



# Target Variable

train_labels = df_train['label'].values

classes = df_train['label'].nunique()

# Encode labels to one hot vectors

train_labels = to_categorical(train_labels, num_classes = classes)

df_train.drop(columns=['label'], inplace=True)



# Spliting the train data

trainX, validX, trainY, validY = train_test_split(df_train, train_labels, test_size=0.2, random_state=42)



# Normalization

trainX = np.array((trainX.astype("float32") / 255.0), dtype='float32')

validX = np.array((validX.astype("float32") / 255.0), dtype='float32')

testX = np.array((df_test.astype("float32") / 255.0), dtype='float32')



# reshape the tuple

trainX = trainX.reshape(trainX.shape[0], *(28, 28, 1))

validX = validX.reshape(validX.shape[0], *(28, 28, 1))

testX = testX.reshape(testX.shape[0], *(28, 28, 1))



# construct the image generator for data augmentation

aug = ImageDataGenerator(featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False,  

        rotation_range=10,  

        zoom_range = 0.1, 

        width_shift_range=0.1,  

        height_shift_range=0.1, 

        horizontal_flip=False,  

        vertical_flip=False)



# instantiate the model

model = MnistModel(classes=classes)



# optimizer

opt = RMSprop(lr=INIT_LR, rho=0.9, epsilon=1e-08, decay=0.0)



# learning rate decay

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



# model compile

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



gc.collect();



# fiting the model through generator

H = model.fit(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),

                        validation_data=(validX, validY),

                        steps_per_epoch=trainX.shape[0] // BATCH_SIZE,

                        epochs=NUM_EPOCHS,

                        callbacks=[learning_rate_reduction],

                        verbose=2)
# Evaluating on validation data

predictions = model.predict(validX, batch_size=BATCH_SIZE)



# determine the number of epochs and then construct the plot title

N = np.arange(0, NUM_EPOCHS)

title = "Training Loss and Accuracy on MNIST ({})".format(model.__class__.__name__)



# plot the training loss and accuracy

plt.style.use("ggplot")

plt.figure()

plt.plot(N, H.history["loss"], label="train_loss")

plt.plot(N, H.history["val_loss"], label="val_loss")

plt.plot(N, H.history["accuracy"], label="train_acc")

plt.plot(N, H.history["val_accuracy"], label="val_acc")

plt.title(title)

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend();
# get the predictions for the test data

predicted_classes = model.predict(testX)



predicted_classes = predicted_classes.argmax(axis=1)
# Plotting the predicted results

L = 3

W = 3

fig, axes = plt.subplots(L, W, figsize = (10,7))

axes = axes.ravel() # 



for i in np.arange(0, L * W):  

    axes[i].imshow(testX[i].reshape(28,28))

    axes[i].set_title("Prediction Class = {:0.1f}".format(predicted_classes[i]))

    axes[i].axis('off')



plt.subplots_adjust(wspace=0.5)
# Submission function

def submission(preds):

    sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

    sub['Label'] = preds

    sub.to_csv('submission.csv', index=False)

    

submission(predicted_classes)