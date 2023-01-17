from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Reshape
class DCGAN:
    @staticmethod
    def build_generator(dim, depth, channels=1, inputDim=100, outputDim=512):

        model = Sequential()

        inputShape = (dim, dim, depth)
        chanDim = -1

        
        model.add(Dense(units=outputDim, input_dim=inputDim))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(Dense(dim * dim * depth))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(Reshape(inputShape))
        model.add(Conv2DTranspose(32, (5,5), strides=(2,2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2DTranspose(channels, (5,5), strides=(2,2), padding="same"))
        model.add(Activation("tanh"))

        return model
    
    @staticmethod
    def build_discriminator(width, height, depth, alpha=0.2):

        model = Sequential()
        inputShape = (height, width, depth)

        model.add(Conv2D(32, (5,5), padding="same", strides=(2,2),
                    input_shape=inputShape))
        model.add(LeakyReLU(alpha=alpha))


        model.add(Conv2D(64, (5,5), padding="same", strides=(2,2)))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        return model
!pip install imutils
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.datasets import mnist
from sklearn.utils import shuffle
from imutils import build_montages
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt
%matplotlib inline
NUM_EPOCHS = 50
BATCH_SIZE = 128

print("[INFO] loading MNIST dataset...")
((trainX, _), (testX, _)) = mnist.load_data()
trainImages = np.concatenate([trainX, testX])

print(trainX.shape)
print(trainImages.shape)

trainImages = np.expand_dims(trainImages, axis=-1)
trainImages = (trainImages.astype("float") - 127.5) / 127.5
print("[INFO] building generator...")
gen = DCGAN.build_generator(7, 64)
print("[INFO] building discriminator...")
disc = DCGAN.build_discriminator(28, 28, 1)
discOpt = Adam(lr=0.0002, beta_1=0.5, decay=0.0002 / NUM_EPOCHS)
disc.compile(loss="binary_crossentropy", optimizer=discOpt)
print("[INFO] building GAN...")
disc.trainable = False
ganInput = Input(shape=(100,))
ganOutput = disc(gen(ganInput))
gan = Model(ganInput, ganOutput)

print(gan.summary())

ganOpt = Adam(lr=0.0002, beta_1=0.5, decay=0.0002 / NUM_EPOCHS)
gan.compile(loss="binary_crossentropy", optimizer=discOpt)
print("[INFO] starting training...")
benchmarkNoise = np.random.uniform(-1, 1, size=(256, 100))
output_dir = os.getcwd()

for epoch in range(NUM_EPOCHS):

    print("[INFO] starting epoch {} of {}...".format(epoch + 1, NUM_EPOCHS))
    batchesPerEpoch = int(trainImages.shape[0] / BATCH_SIZE)

    for i in range(0, batchesPerEpoch):

        p = None

        imageBatch = trainImages[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

        genImages = gen.predict(noise, verbose=0)

        X = np.concatenate((imageBatch, genImages))
        y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
        (X, y) = shuffle(X, y)

        discLoss = disc.train_on_batch(X, y)
        
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        ganLoss = gan.train_on_batch(noise, [1] * BATCH_SIZE)

        if i == batchesPerEpoch - 1:
            p = [output_dir, "epoch_{}_output.png".format(str(epoch + 1).zfill(4))]

        if p is not None:
            print("[INFO] Step {}_{}: discriminator_loss={:.6f}, "
                "adversarial_loss={:.6f}".format(epoch+1, i, discLoss, ganLoss))

            images = gen.predict(benchmarkNoise)
            images = ((images * 127.5) + 127.5).astype("uint8")
            images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (28,28), (16,16))[0]

            p = os.path.sep.join(p)
            cv2.imwrite(p, vis)
!ls
out = cv2.imread("epoch_0050_output.png")
plt.axis("off")
plt.imshow(out)