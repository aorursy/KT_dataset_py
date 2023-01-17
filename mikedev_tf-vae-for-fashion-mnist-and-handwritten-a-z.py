# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from math import ceil
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!mkdir logdir
!mkdir ckpts
def tf_namespace(namespace):
    def wrapper(f):
        def wrapped_f(*args, **kwargs):
            with tf.name_scope(namespace):
                return f(*args, **kwargs)

        return wrapped_f

    return wrapper


class ImgSaver:

    def __init__(self, n, w, h, prefix=None):
        self.w = w
        self.h = h
        self.n = n
        self.prefix = prefix or 'img'

    def set_vae(self, vae):
        self.vae = vae

    def __call__(self, epoch=None):
        samples = self.vae.generate(self.n).reshape(-1, self.w, self.h)
        suffix = '' if epoch is None else f'_epoch_{epoch}'
        for i in range(self.n):
            sample = samples[i, :, :]
            plt.imsave(f'{self.prefix}_{i}' + suffix + '.jpg', sample, cmap='gray')

@tf_namespace('psigmoid')
def psigmoid(x, beta=None, alpha=None, beta_start=1., alpha_start=0.):
    if beta is None:
        beta = tf.Variable(beta_start, name='beta', dtype=tf.float32)
    if alpha is None:
        alpha = tf.Variable(alpha_start, name='alpha', dtype=tf.float32)
    exponent = (x + alpha) * beta
    e_x = tf.exp(exponent)
    return tf.divide(e_x,(e_x+1.), 'psigmoid')


class VAE:

    def __init__(self, input_shape, encode_sizes, latent_size, decode_sizes=None, mu_prior=None, sigma_prior=None, lr=10e-4,
                 momentum=0.9, n_saver=20, prefix_imsaver=None, model_name=None):
        self.encode_sizes = encode_sizes
        self.latent_size = latent_size
        self.decode_sizes = decode_sizes or encode_sizes[::-1]
        self.model_name = model_name or 'mikedev_vae'
        self.mu_prior = mu_prior or np.zeros([latent_size], dtype='float32')
        self.sigma_prior = sigma_prior or np.ones([latent_size], 'float32')
        self.lr = lr
        self.momentum = momentum
        self.input_shape = input_shape
        self._build_graph(input_shape, latent_size)
        self.imsaver = ImgSaver(20, 28, 28, prefix_imsaver)

    def _build_graph(self, input_shape, latent_size):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._create_placeholders(input_shape)
            self._create_encoder(self.X)
            self._create_latent_distribution(self.encoder, latent_size)
            self._create_decoder(self.z)
            self.loss = - self.elbo(self.X, self.decoder_logits, self.mu, self.log_sigma_square, self.sigma_square,
                                    tf.constant(self.mu_prior), tf.constant(self.sigma_prior))
            self.opt = tf.train.AdamOptimizer(self.lr, self.momentum)
            self.opt_op = self.opt.minimize(self.loss)
            self.session = tf.InteractiveSession(graph=self.graph)
        writer = tf.summary.FileWriter(logdir='logdir', graph=self.graph)
        writer.flush()

    @property
    def k_init(self):
        return {'kernel_initializer': tf.glorot_uniform_initializer()}

    def elbo(self, X_true, X_logits, mu, log_sigma, sigma, mu_prior, sigma_prior):
        epsilon = tf.constant(0.000001)
        mae = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(X_true, X_logits, reduction=tf.losses.Reduction.NONE), axis=1)
        log_sigma_prior = tf.log(sigma_prior + epsilon)
        mu_diff = mu - mu_prior
        self.kl = 0.5 * tf.reduce_sum(log_sigma_prior - log_sigma - 1 +
                                      (sigma + tf.multiply(mu_diff, mu_diff)) / sigma_prior, axis=1)
        return  tf.reduce_mean( - mae - self.kl)

    @tf_namespace('placeholders')
    def _create_placeholders(self, input_shape):
        self.X = tf.placeholder(tf.float32, shape=[None, *input_shape], name='X')

    @tf_namespace('encoder')
    def _create_encoder(self, X):
        self.encode_layers = []
        self.encoder = X
        self.encoder = tf.reshape(self.encoder, [tf.shape(self.encoder)[0], 28, 28, 1])
        for i, conv_size in enumerate(self.encode_sizes):
            self.encoder = tf.layers.conv2d(self.encoder, *conv_size, **self.k_init,
                                           activation=tf.nn.relu, name=f'encoder_{i+1}')
            self.encode_layers.append(self.encoder)
            setattr(self, f'encoder_{i + 1}', self.encoder)

    @tf_namespace('latent')
    def _create_latent_distribution(self, encoder, latent_dim):
        self.flatter_layer = tf.layers.flatten(encoder)
        self.mu = tf.layers.dense(self.flatter_layer, latent_dim, **self.k_init, name='mu')
        self.log_sigma_square = tf.layers.dense(self.flatter_layer, latent_dim,
                                                **self.k_init, name='log_sigma_square')
        self.sigma_square = tf.exp(self.log_sigma_square, 'sigma_square')
        self.z = tf.add(self.mu, self.sigma_square * tf.random.normal(tf.shape(self.sigma_square)), 'z')

    @tf_namespace('decoder')
    def _create_decoder(self, z):
        self.decoder = z
        self.decode_layers = []
        for i, conv_size in enumerate(self.decode_sizes):
            self.decoder = tf.layers.dense(self.decoder, conv_size, **self.k_init,
                                           activation=tf.nn.relu, name=f'decoder_{i+1}')
            setattr(self, f'decoder_{i + 1}', self.decoder)
            self.decode_layers.append(self.decoder)
            if i == len(self.decode_sizes)-1:
                self.decoder = tf.layers.dense(self.decoder, self.input_shape[0],
                                               **self.k_init, name=f'decoder_{i+2}')
                self.decoder_logits = self.decoder
                self.decoder = psigmoid(self.decoder, beta=3., alpha=0.)
                setattr(self, f'decoder_{i+2}', self.decoder)
                self.decode_layers.append(self.decoder)
        return self.decoder

    @property
    def layers(self):
        return [(f'encoder_{i}', getattr(self, f'encoder_{i}')) for i in range(1, len(self.encode_layers) + 1)] + \
               [('flatten', self.flatter_layer), ('mu', self.mu), ('sigma', self.log_sigma_square), ('z', self.z)] + \
               [(f'decoder_{i}', getattr(self, f'decoder_{i}')) for i in range(1, len(self.decode_layers) + 1)]

    def fit(self, X, epochs, batch_size, print_every=50, save_every=10):
        if self.imsaver is not None:
            self.imsaver.set_vae(self)
        n_batch = ceil(X.shape[0] / batch_size)
        saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())
        self.history = []
        for epoch in range(1, epochs + 1):
            np.random.shuffle(X)
            acc_loss = 0
            counter = 0
            epoch_hs = []
            for i in range(n_batch):
                slice_batch = slice(i * batch_size, (i + 1) * batch_size) if i != n_batch - 1 else slice(
                    i * batch_size,
                    None)
                X_batch = X[slice_batch, :]
                batch_loss, _ = self.session.run([self.loss, self.opt_op], {self.X: X_batch})
                acc_loss += batch_loss
                epoch_hs.append(batch_loss)
                if counter % print_every == 0:
                    print(f" Epoch {epoch} - batch {i} - neg_ELBO = {batch_loss}")
                counter += 1
            print(f'\nEpoch {epoch} - Avg loss = {acc_loss / n_batch}')
            print('\n' + ('-' * 70))
            self.history.append(epoch_hs)
            saver.save(self.session, f"ckpts/{self.model_name}.ckpt")
            if self.imsaver is not None and epoch % save_every == 0:
                self.imsaver(epoch)

    def generate(self, n=1, mu_prior=None, sigma_prior=None):
        if mu_prior is None:
            mu_prior = self.mu_prior
        if sigma_prior is None:
            sigma_prior = self.sigma_prior
        z = np.random.multivariate_normal(mu_prior, np.diag(sigma_prior), [n])
        return self.session.run(self.decoder, feed_dict={self.z: z})

    def reconstruct(self, X):
        return self.session.run(self.decoder, feed_dict={self.X: X})

    def open(self):
        if not hasattr(self, 'session') or self.session is None:
            if self.graph is None:
                self._build_graph(self.input_shape, self.latent_size)
            else:
                self.session = tf.InteractiveSession(graph=self.graph)

    def close(self):
        if hasattr(VAE, 'session') and VAE.session is not None:
            VAE.session.close()
            VAE.session = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __delete__(self, instance):
        self.close()

    def __setattr__(self, key, value):
        if key == 'session':
            if hasattr(self, 'session') and self.session is not None:
                self.session.close()
            elif hasattr(VAE, 'session') and VAE.session is not None:
                VAE.session.close()
            VAE.session = value
        else:
            self.__dict__[key] = value

    def __delattr__(self, item):
        if item == 'session':
            self.close()
            del VAE.__dict__['session']
        else:
            del self.__dict__[item]

    def __enter__(self):
        self.open()
        return self
def plot_generated_images(mikedev_vae, n = 15):
    X_new = mikedev_vae.generate(n).reshape((n, 28, 28))
    for i in range(n):
        plt.imshow(X_new[i, :, :], 'gray')
        plt.imsave(f'pic_{i}_epoch_100.jpg', X_new[i, :, :], cmap='gray')
        plt.show()
def plot_reconstructed_images(X_train, mikedev_vae, n=15):
    ix = np.random.randint(0, X_train.shape[0], n)
    X_orig = X_train[ix, :]
    X_new = mikedev_vae.reconstruct(X_orig).reshape((n, 28, 28))
    for i in range(n):
        fig, axs= plt.subplots(1, 2)
        ax1, ax2 = axs
        ax1.imshow(X_orig[i, :].reshape((28, 28)), 'gray')
        ax2.imshow(X_new[i, :, :], 'gray')
        plt.show()
train = pd.read_csv('../input/handwritten-az/handwritten_data_785.csv')
train.columns = ['label'] + [f'pixel_{i}' for i in range(1, 785)]
train = train.sample(100_000, random_state=13)
X_train, y_train = train.drop(columns='label').values, train['label'].values
X_train = X_train.astype('float32')
X_train = X_train / 255
encoder = [(8, 3), (16,5), (32,5), (32,5)]
decoder = [57, 124, 353, 674]
with VAE((784,), encoder, 2, decoder, prefix_imsaver='char', model_name='char_vae') as mikedev_vae:
    mikedev_vae.fit(X_train, epochs=100, batch_size=256, print_every=100000)
    with open('char_history_training', 'w') as f:
        f.write(str(mikedev_vae.history))
    plot_generated_images(mikedev_vae)
    plot_reconstructed_images(X_train, mikedev_vae)
train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
X_train, y_train = train.drop(columns='label').values, train['label'].values
X_train = X_train.astype('float32')
X_train = X_train / 255
encoder = [(8, 3), (16,5), (32,5), (32,5)]
decoder = [57, 124, 353, 674]
with VAE((784,), encoder, 2, decoder, prefix_imsaver='fashion', model_name='fashion_vae') as mikedev_vae:
    mikedev_vae.fit(X_train, epochs=100, batch_size=256, print_every=100000)
    with open('fashion_history_training', 'w') as f:
        f.write(str(mikedev_vae.history))
    plot_generated_images(mikedev_vae)
    plot_reconstructed_images(X_train, mikedev_vae)