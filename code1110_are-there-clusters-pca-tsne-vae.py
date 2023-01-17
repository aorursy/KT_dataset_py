# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import warnings

warnings.filterwarnings("ignore")



import cv2

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.cm import get_cmap

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split



from scipy.stats import norm



import keras

from keras import layers

from keras.models import Model

from keras import metrics

from keras import backend as K   # 'generic' backend so 
# load train and test.csv

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# How does the train file look like?

train.head()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.countplot(train['diagnosis'], ax=ax)
# How does the test file look like?

test.head()
# train images

train_image_list = os.listdir("../input/train_images")

print("{} train images".format(len(train_image_list)))

print(train_image_list[:10])
# test images

test_image_list = os.listdir("../input/test_images")

print("{} test images".format(len(test_image_list)))

print(test_image_list[:10])
# visualize random N images whose score is 0 (No DR)

def visualize_images(train, s, N):

    # select image with the score of s

    train_image_list = train.loc[train['diagnosis'] == s, 'id_code'].tolist()    

    print("There are {} / 3662 images with the score of {}.".format(len(train_image_list), s))



    # random N choices of images

    np.random.seed(42)

    random_choice = np.random.choice(len(train_image_list), N)

    

    # visualize

    nrow = int(np.floor(np.sqrt(N)))

    fig, ax = plt.subplots(nrow, int(N/nrow), figsize=(20, 12))

    ax = ax.flatten()

    for n in range(N):

        img = cv2.imread('../input/train_images/' + train_image_list[random_choice[n]] + '.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB

        img = cv2.resize(img, (512, 512)) # resize to 512 x 512 (for now)

        ax[n].imshow(img)

        ax[n].set_title(train_image_list[random_choice[n]])    

    plt.tight_layout()

    

# examples of images (score = 0) 

visualize_images(train, 0, 30)
# visualize random N images whose score is 1 (mild)

visualize_images(train, 1, 30)
# visualize random N images whose score is 1 (mild)

visualize_images(train, 2, 30)
# visualize random N images whose score is 1 (mild)

visualize_images(train, 3, 30)
# visualize random N images whose score is 4 (Proliferative DR)

visualize_images(train, 4, 30)
# train image data

npix = 224 # resize to npix x npix (for now)

X_train = np.zeros((train.shape[0], npix, npix))

for i in range(train.shape[0]):

    # load an image

    img = cv2.imread('../input/train_images/' + train.loc[i, 'id_code'] + '.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    X_train[i, :, :] = cv2.resize(img, (npix, npix)) 

    

print("X_train shape: " + str(np.shape(X_train)))   
# test image data

X_test = np.zeros((test.shape[0], npix, npix))

for i in range(test.shape[0]):

    # load an image

    img = cv2.imread('../input/test_images/' + test.loc[i, 'id_code'] + '.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    X_test[i, :, :] = cv2.resize(img, (npix, npix)) 

    

print("X_test shape: " + str(np.shape(X_test))) 
def laplacian_variance(images):

    return [cv2.Laplacian(image, cv2.CV_32F).var() for image in images]



laplacian_variances = laplacian_variance(X_train.astype('uint8'))

# print(len(laplacian_variances))

# print(len(train['diagnosis']))

blur_df = pd.DataFrame({'diagnosis': train['diagnosis'], 'laplacian_variance': laplacian_variances})



fig, ax = plt.subplots(1, 1, figsize=(12, 8))

sns.boxplot(x='diagnosis', y='laplacian_variance', data=blur_df, ax=ax)

ax.set_title('blurriness')
# normalize

X = X_train / 255



# reshape

X = X.reshape(X.shape[0], -1)



# use a subset of the data

trainX, valX, trainy, valy = train_test_split(X, train['diagnosis'], test_size=0.5, random_state=1220)



X_decomposed = PCA(n_components=2).fit_transform(trainX)



fig, ax = plt.subplots(1, 1, figsize=(12, 8))

cmap = get_cmap("tab10")



for i in range(5):

    marker = "$" + str(i) + "$"

    idx = trainy == i

    ax.scatter(X_decomposed[idx, 0], X_decomposed[idx, 1],

              marker=marker, color=cmap(i))

ax.set_title("PCA")
X_decomposed = TSNE(n_components=2).fit_transform(trainX)



fig, ax = plt.subplots(1, 1, figsize=(12, 8))



for i in range(5):

    marker = "$" + str(i) + "$"

    idx = trainy == i

    ax.scatter(X_decomposed[idx, 0], X_decomposed[idx, 1],

              marker=marker, color=cmap(i))

ax.set_title("t-SNE")
# hyperparameters

epochs = 10

batch_size = 200

latent_dim = 2 # dimensionality of the latent space



# encoder

input_img = keras.Input(shape=(npix, npix, 1))

x = layers.Conv2D(32, 3,

                  padding='same', 

                  activation='relu')(input_img)

x = layers.Conv2D(64, 3,

                  padding='same', 

                  activation='relu',

                  strides=(2, 2))(x)

x = layers.Conv2D(64, 3,

                  padding='same', 

                  activation='relu')(x)

x = layers.Conv2D(64, 3,

                  padding='same', 

                  activation='relu')(x)

# need to know the shape of the network here for the decoder

shape_before_flattening = K.int_shape(x)



x = layers.Flatten()(x)

x = layers.Dense(32, activation='relu')(x)



# Two outputs, latent mean and (log)variance

z_mu = layers.Dense(latent_dim)(x)

z_log_sigma = layers.Dense(latent_dim)(x)
# sampling function

def sampling(args):

    z_mu, z_log_sigma = args

    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),

                              mean=0., stddev=1.)

    return z_mu + K.exp(z_log_sigma) * epsilon



# sample vector from the latent distribution

z = layers.Lambda(sampling)([z_mu, z_log_sigma])
# decoder

decoder_input = layers.Input(K.int_shape(z)[1:])



# expand to npix x npix pixels

x = layers.Dense(np.prod(shape_before_flattening[1:]),

                 activation='relu')(decoder_input)



# reshape

x = layers.Reshape(shape_before_flattening[1:])(x)



# use Conv2DTranspose to reverse the conv layers from the encoder

x = layers.Conv2DTranspose(32, 3, padding='same', 

                           activation='relu', strides=(2, 2))(x)

x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)



# declare decoder

decoder = Model(decoder_input, x)



# apply the decoder to the sample from the latent distribution

z_decoded = decoder(z)
# loss

class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):

        x = K.flatten(x)

        z_decoded = K.flatten(z_decoded)

        

        # reconstruction loss

        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)

        

        # KL divergence

        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)

        

        return K.mean(xent_loss + kl_loss)

    

    # adds the custom loss to the class

    def call(self, inputs):

        x = inputs[0]

        z_decoded = inputs[1]

        loss = self.vae_loss(x, z_decoded)

        self.add_loss(loss, inputs=inputs)

        return x

    

# apply the custom loss to the input images and the decoded latent distribution sample

y = CustomVariationalLayer()([input_img, z_decoded])
# VAE compile

vae = Model(input_img, y)

vae.compile(optimizer='rmsprop', loss=None)
# train VAE

trainX = np.reshape(trainX, (-1, npix, npix, 1))

valX = np.reshape(valX, (-1, npix, npix, 1))

# print(np.shape(valX))

# print(np.shape(trainX))

vae.fit(x=trainX, y=None, 

        shuffle=True, epochs=epochs,

        batch_size=batch_size, 

        validation_data=(valX, None))
# translate into the latent space

encoder = Model(input_img, z_mu)

X_encoded = encoder.predict(valX, batch_size=batch_size)



# plot

fig, ax = plt.subplots(1, 1, figsize=(12, 8))



for i in range(5):

    marker = "$" + str(i) + "$"

    idx = valy == i

    ax.scatter(X_encoded[idx, 0], X_encoded[idx, 1],

              marker=marker, color=cmap(i))

ax.set_title("VAE's latent space")
# Display a 2D manifold of the digits

n = 20  # figure with 20x20 digits

digit_size = npix

figure = np.zeros((digit_size * n, digit_size * n))



# Construct grid of latent variable values

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))

grid_y = norm.ppf(np.linspace(0.05, 0.95, n))



# decode for each square in the grid

for i, yi in enumerate(grid_x):

    for j, xi in enumerate(grid_y):

        z_sample = np.array([[xi, yi]])

        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)

        x_decoded = decoder.predict(z_sample, batch_size=batch_size)

        digit = x_decoded[0].reshape(digit_size, digit_size)

        figure[i * digit_size: (i + 1) * digit_size,

               j * digit_size: (j + 1) * digit_size] = digit



fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax.imshow(figure, cmap='gnuplot2')

ax.set_title('generated images by VAE') 