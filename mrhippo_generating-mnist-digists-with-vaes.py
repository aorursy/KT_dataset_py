# data processing
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# keras
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras import objectives
import keras.backend as K
import keras

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
data.head()
data_target = data["label"]
data = data.drop(["label"], axis = 1) # drop label
train, test, _, test_labels = train_test_split(data, data_target, test_size = 0.2, random_state = 42)
train = train.to_numpy()
test = test.to_numpy()
# normalization
train = train / 255
test = test / 255
def show_samples(data):

    fig, ax = plt.subplots(4, 5, figsize = (15, 8))
    index = 1

    for column in range(0, 5):
        for row in range(0, 4):
            ax[row, column].imshow(data[index].reshape(28,28), cmap = "Greys_r")
            ax[row, column].axis(False)
            index += 200

    plt.show()
show_samples(train)
LATENT_SPACE = 2
BATCH_SIZE = 100 # 256 gives an error (I don't know why, if you know please let me know in the comments)
input_img = Input(shape = (784,))

encoded = Dense(256, activation = "relu")(input_img)
encoded = Dense(128, activation = "relu")(encoded)

mu = Dense(LATENT_SPACE)(encoded)
sigma = Dense(LATENT_SPACE)(encoded)
def sampling(args):
    mu, sigma = args
    eps = K.random_normal(shape = (BATCH_SIZE, LATENT_SPACE), mean=0., stddev=1.0)
    return mu + K.exp(sigma) * eps

z = Lambda(sampling, output_shape=(LATENT_SPACE,))([mu, sigma])
decoder1 = Dense(128, activation = "relu")
decoder2 = Dense(256, activation = "relu")
decoder3 = Dense(784, activation = "sigmoid")

z_decoded = decoder1(z)
z_decoded = decoder2(z_decoded)
y = decoder3(z_decoded)
reconstruction_loss = objectives.binary_crossentropy(input_img, y) * train.shape[1]
kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(sigma) - sigma - 1, axis = -1)
vae_loss = reconstruction_loss + kl_loss

vae = Model(input_img, y)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()
hist = vae.fit(train,
       shuffle=True,
       epochs=20,
       batch_size=BATCH_SIZE,
       validation_data=(test, None))
plt.figure(figsize = (15, 6))
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Test Loss")
plt.title("Losses")
plt.legend()
plt.show()
encoder = Model(input_img, mu)
encoder.summary()
test_latent = encoder.predict(test, batch_size=BATCH_SIZE)
plt.figure(figsize=(10,10))
plt.scatter(test_latent[:, 0], test_latent[:, 1], c=test_labels)
plt.title("Label Cluesters")
plt.colorbar()
plt.show()
decoder_input = Input(shape = (LATENT_SPACE,))
decoder = decoder1(decoder_input) # ! 
decoder = decoder2(decoder)
new_decoder = decoder3(decoder)

generator = Model(decoder_input, new_decoder)
generator.summary()
# display a 2D manifold of the digits
n = 15 # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) 
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()