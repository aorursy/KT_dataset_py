import numpy as np
import matplotlib.pyplot as plt
# Ensure results are reproducable
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
def mnist_load_data(path='mnist.npz'):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)
        

(X_train, y_train), (X_test, y_test) = mnist_load_data(path='../input/mnist.npz')
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
plt.imshow(X_test[0])
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
X_train_flat = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test_flat = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
print(X_train_flat.shape)
print(X_test_flat.shape)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X_train_flat, X_train_flat,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test_flat, X_test_flat))
original = np.expand_dims(X_test_flat[0],0)
seven = autoencoder.predict(original)
seven = seven.reshape(1,28,28)
original = original.reshape(1,28,28)
fig = plt.figure(figsize=(7, 10))
a=fig.add_subplot(1,2,1)
a.set_title('Original')
imgplot = plt.imshow(original[0,:,:])

b=fig.add_subplot(1,2,2)
b.set_title('Autoencoder')
imgplot = plt.imshow(seven[0,:,:])
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
batch_size = 100
original_dim = 784
latent_dim = 32
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)
# instantiate VAE model
vae = Model(x, x_decoded_mean)
# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()
vae.fit(X_train_flat,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_flat, None))
one_seven = X_test_flat[0]
one_seven = np.expand_dims(one_seven,0)
one_seven.shape
one_seven = one_seven.repeat(32,axis=0)
s = vae.predict(one_seven)
s.shape
s = s.reshape(128,28,28)
smean = s.mean(axis=0)
plt.imshow(smean)
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)
# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()
# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)
generator.summary()
y_test.shape
sevens = X_test_flat[np.where(y_test == 7)]
sevens.shape
enc = encoder.predict(X_test_flat)
enc_mean = enc.mean(axis=0,keepdims=True)
g = generator.predict(enc_mean)
g.shape
g = g.reshape(g.shape[0],28,28)
plt.imshow(g[0,:,:])
from sklearn.manifold import TSNE
tsne = TSNE(verbose=1, n_iter = 300)
enc.shape
man = tsne.fit_transform(enc)
man.shape
fig = plt.figure(figsize=(10,7))
scatter =plt.scatter(man[:,0],man[:,1],c=y_test,cmap='hsv', s=0.6)
scatter.axes.get_xaxis().set_visible(False)
scatter.axes.get_yaxis().set_visible(False)