
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
%matplotlib notebook

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import keras 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_tf import model_eval, model_argmax
from cleverhans.train import train
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import BasicIterativeMethod
from sklearn.model_selection import train_test_split
from cleverhans.dataset import MNIST

tf.reset_default_graph()
# Object used to keep track of (and return) key accuracies
report = AccuracyReport()

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

# Create TF session and set as Keras backend session
sess = tf.Session()

set_log_level(logging.DEBUG)

# Get MNIST test data
mnist = MNIST(train_start=0, train_end=60000,
            test_start=0, test_end=10000)
x_train, y_train = mnist.get_set('train')
x_test, y_test = mnist.get_set('test')

# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                    nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

nb_filters = 64
# Define TF model graph
model = ModelBasicCNN('model1', nb_classes, nb_filters)
preds = model.get_logits(x)
loss = CrossEntropy(model, smoothing=0.1)
print("Defined TensorFlow model graph.")

###########################################################################
# Training the model using TensorFlow
###########################################################################
NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
SOURCE_SAMPLES = 10

# Train an MNIST model
train_params = {
  'nb_epochs': NB_EPOCHS,
  'batch_size': BATCH_SIZE,
  'learning_rate': LEARNING_RATE
}
sess.run(tf.global_variables_initializer())
rng = np.random.RandomState([2017, 8, 30])
train(sess, loss, x_train, y_train, args=train_params, rng=rng)

# Evaluate the accuracy of the MNIST model on legitimate test examples
eval_params = {'batch_size': BATCH_SIZE}
accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
assert x_test.shape[0] == 10000 , x_test.shape
print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
def get_bim_params(clip_min=0.,clip_max=1., eps_iter = 0.01, nb_iter= 100):
    bim_params = {'eps_iter': eps_iter,
                  'nb_iter': nb_iter,
                  'clip_min': clip_min,
                  'clip_max': clip_max}
    return bim_params

bim_op = BasicIterativeMethod(model, sess=sess)

advs = bim_op.generate_np(x_train[:10000], **get_bim_params(nb_iter=20, eps_iter=0.01))
y_train.shape
x_train_advs=np.append(x_train,advs).reshape(70000,784)
y_train_advs= np.append(np.argmax(y_train,axis=1),10*np.ones((10000,1)))
y_train_advsCat= keras.utils.to_categorical(y_train_advs)
x_train_advs.shape
res_advs = np.array(model_argmax(sess, x, preds, advs) ==np.argmax(y_train[:10000], axis=1)).astype(int)
accuracy_advs=res_advs.sum()/len(res_advs)
print(accuracy_advs)
y_train_advs.shape
X_train_, X_test_, y_train_, y_test_ = train_test_split(x_train_advs, y_train_advs, random_state=0)
#tf.reset_default_graph()


'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''



# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
#x_train = np.reshape(x_train, [-1, original_dim])
#x_test = np.reshape(x_test, [-1, original_dim])
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 5

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x2 = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x2)
z_log_var = Dense(latent_dim, name='z_log_var')(x2)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x2 = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x2)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    help_ = "Load h5 model trained weights"
#    parser.add_argument("-w", "--weights", help=help_)
#    help_ = "Use mse loss instead of binary cross entropy (default)"
#    parser.add_argument("-m",
#                        "--mse",
#                        help=help_, action='store_true')
 #   args = parser.parse_args()
    models = (encoder, decoder)
    data = (X_test_, y_test_)

    # VAE loss = mse_loss or xent_loss + kl_loss
    #if args.mse:
    #    reconstruction_loss = mse(inputs, outputs)
    #else:
    reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    # train the autoencoder
    vae.fit(X_train_,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_.reshape(X_test_.shape[0],784), None))
    #vae.save_weights('vae_mlp_mnist.h5')

plot_results(models,
                 (x_train_advs, y_train_advs),
                 batch_size=batch_size,
                 model_name="vae_mlp")

numberofVaeAdvs=10
grid_x = np.linspace(0., 1, numberofVaeAdvs)
grid_y = np.linspace(0.5, 1.8, numberofVaeAdvs)[::-1]
vae_advs=[]
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(28, 28)
        vae_advs.append(digit)

pred_vae_advs =model_argmax(sess, x, preds, np.array(vae_advs).reshape(100,28,28,1))
plt.figure(figsize=(15,20))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.title("predicting {}".format(pred_vae_advs[i]))
    plt.imshow(vae_advs[i])
    
plt.show()
z_sample = np.array([[2., -2.]])
x_decoded = decoder.predict(z_sample)
pred_vae_adv=model_argmax(sess, x, preds, np.array(x_decoded).reshape(1,28,28,1))
plt.figure()
plt.imshow(x_decoded.reshape(28,28))
plt.title("Generating vae image from {} predicting {}".format(z_sample[0],pred_vae_adv))
plt.show()
