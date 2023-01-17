from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import Conv2DTranspose 

from tensorflow.keras.layers import LeakyReLU 

from tensorflow.keras.layers import Activation 

from tensorflow.keras.layers import Flatten 

from tensorflow.keras.layers import Dense 

from tensorflow.keras.layers import Reshape 

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model

from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt

import numpy as np
def ConvAutoEncoder(width, height, depth, filters=(32, 64), latentDim=16):



  input_shape = (width, height, depth)

  chanDim = -1



  encInp = Input(shape=input_shape)

  x = encInp



  for f in filters:

    x = Conv2D(f, (3, 3), strides=2, padding='same')(x)

    x = LeakyReLU(alpha=0.2)(x)

    x = BatchNormalization(axis=chanDim)(x)



  volumeSize = K.int_shape(x)

  x = Flatten()(x)

  latent = Dense(latentDim)(x)



  encoder = Model(encInp, latent, name='encoder')



  decInp = Input(shape=(latentDim, ))

  x = Dense(np.prod(volumeSize[1:]))(decInp)

  x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)



  for f in filters[::-1]:

    x = Conv2DTranspose(f, (3, 3), strides=2, padding='same')(x)

    x = LeakyReLU(alpha=0.2)(x)

    x = BatchNormalization(axis=chanDim)(x)



  x = Conv2DTranspose(depth, (3, 3), padding='same')(x)

  output = Activation('sigmoid')(x)



  decoder = Model(decInp, output, name='decoder')



  autoEncoder = Model(encInp, decoder(encoder(encInp)), name='autoencoder')



  return (encoder, decoder, autoEncoder)
data = mnist.load_data()

(trainX, _), (testX, _) = mnist.load_data()



trainX = np.expand_dims(trainX, axis=-1)

testX = np.expand_dims(testX, axis=-1)

trainX = trainX.astype("float32") / 255.0

testX = testX.astype("float32") / 255.0
trainNoise = np.random.uniform(low=-800, high=800, size=trainX.shape)/1000.0

testNoise = np.random.uniform(low=-800, high=800, size=testX.shape)/1000.0

trainXNoisy = np.clip(trainX + trainNoise, 0, 1)

testXNoisy = np.clip(testX + testNoise, 0, 1)
test_set_len = len(testXNoisy)

split_at = int(0.7*test_set_len)



validXNoisy = testXNoisy[:split_at]

testXNoisy = testXNoisy[split_at:]



validX = testX[:split_at]

testX = testX[split_at:]
(encoder, decoder, autoencoder) = ConvAutoEncoder(28, 28, 1)
plot_model(autoencoder)
encoder.summary()
decoder.summary()
opt = Adam(lr=1e-3)

autoencoder.compile(loss="mse", optimizer=opt)
EPOCHS = 25

BS = 32



H = autoencoder.fit(

    trainXNoisy, trainX,

    validation_data=(validXNoisy, validX),

    epochs=EPOCHS,

    batch_size=BS)
fontdict = {'weight':'bold', 'fontsize':14}

plt.figure(figsize=(8, 5))

plt.plot(H.history['loss'], label='trainLoss', linewidth=2)

plt.plot(H.history['val_loss'], label='valLoss', linewidth=2)

plt.xlabel('Epoch #', fontdict=fontdict )

plt.ylabel('Loss #', fontdict=fontdict )

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.legend()

plt.grid()

plt.show()
def plot_samples(n_samples, noisy_data):



  total_samples = len(noisy_data)

  sample_inds = np.random.choice([i for i in range(total_samples)], n_samples)



  y_hat = autoencoder.predict(noisy_data[sample_inds]).reshape(-1, 28, 28)

  y = noisy_data[sample_inds].reshape(-1, 28, 28)



  fig, ax = plt.subplots(n_samples, 2, figsize=(1.95,n_samples), gridspec_kw = {'wspace':0, 'hspace':0})

  for i in range(n_samples):

    ax[i, 0].imshow(y[i], 'gray')

    ax[i, 1].imshow(y_hat[i], 'gray')

    ax[i, 0].axis('off')

    ax[i, 1].axis('off')
plot_samples(8, testXNoisy)