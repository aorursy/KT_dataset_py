%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams["figure.figsize"] = (15, 10)

plt.rcParams["figure.dpi"] = 300

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

plt.rcParams['image.cmap'] = 'gray' # grayscale looks better

from itertools import cycle

prop_cycle = plt.rcParams['axes.prop_cycle']

colors = prop_cycle.by_key()['color']
from pathlib import Path

import numpy as np

import pandas as pd

import os

from skimage.io import imread, imsave

from IPython.display import clear_output

from skimage.util import montage

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

from skimage.color import label2rgb

import h5py
with h5py.File('../input/packaging-images-and-features/out_cartoons.h5', 'r') as h:

    val_dict = {}

    for k in h.keys():

        print(k, h[k].shape)

        val_dict[k] = list(h[k])

    cartoon_df = pd.DataFrame(val_dict)

    del val_dict

cartoon_df.sample(3)
val_count = {}

for c_col in cartoon_df.columns:

    if c_col not in ['image']:

        val_count[c_col] = cartoon_df[c_col].value_counts().index.max()+1

print(sum(val_count.values()), 'states')

val_count
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(cartoon_df, 

                                     random_state=0, 

                                     test_size=0.25) # stratify=cartoon_df[val_count.keys()],
from keras import layers, models

from keras.losses import mse, binary_crossentropy

from keras.utils import plot_model as pm_raw

def plot_model(*args, **kwargs):

    try:

        return pm_raw(*args, **kwargs)

    except Exception as e:

        print(e)

from keras import backend as K
original_shape = train_df['image'].shape[1:]

original_dim = np.prod(original_shape)



# parameters

ARGS_MSE = False

EPOCHS = 100

BATCH_SIZE = 64

TARGET_DIM_SIZE = 128

TARGET_LATENT_SIZE = 2

recon_loss_func = mse if ARGS_MSE else binary_crossentropy
# reparameterization trick

# instead of sampling from Q(z|X), sample epsilon = N(0,I)

# z = z_mean + sqrt(var) * epsilon



def sampling(args):

    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments

        args (tensor): mean and log of variance of Q(z|X)

    # Returns

        z (tensor): sampled latent vector

    """



    z_mean, z_log_var = args

    batch = K.shape(z_mean)[0]

    dim = K.int_shape(z_mean)[1]

    # by default, random_normal has mean = 0 and std = 1.0

    epsilon = K.random_normal(shape=(batch, dim))

    return z_mean + K.exp(0.5 * z_log_var) * epsilon





def z_log_loss(_, x):

    return K.exp(x)-x





def z_mean_loss(_, x):

    return K.square(x)
def build_dcgan_vae(input_shape,

                    intermediate_dim=TARGET_DIM_SIZE,

                    latent_dim=TARGET_LATENT_SIZE,

                    cnn_blocks=5,

                    cnn_depth=8):

    # VAE model = encoder + decoder

    # build encoder model

    # dcgan style CNN

    raw_inputs = layers.Input(shape=input_shape, name='encoder_input')

    cur_x = raw_inputs

    for i in range(cnn_blocks):

        cur_x = layers.Conv2D(cnn_depth*2**i,

                              (3, 3),

                              activation='linear',

                              padding='same',

                              strides=(2, 2),

                              use_bias=False)(cur_x)

        cur_x = layers.BatchNormalization()(cur_x)

        cur_x = layers.LeakyReLU(0.2)(cur_x)

    inputs = layers.Flatten()(cur_x)

    x = layers.Dense(intermediate_dim, activation='relu')(inputs)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)

    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)



    # use reparameterization trick to push the sampling out as input

    # note that "output_shape" isn't necessary with the TensorFlow backend

    z = layers.Lambda(sampling, output_shape=(

        latent_dim,), name='z')([z_mean, z_log_var])



    # instantiate encoder model

    encoder = models.Model(raw_inputs, [z_mean, z_log_var, z], name='encoder')

    plot_model(encoder, to_file='vae_dcgan_encoder.png', show_shapes=True)



    # build decoder model

    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')

    x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)

    int_shape = (input_shape[0]//2**cnn_blocks,

                 input_shape[1]//2**cnn_blocks,

                 cnn_depth*2**cnn_blocks)

    ds_features = layers.Dense(np.prod(int_shape).astype(int),

                               activation='relu', name='decoding_blcok')(x)

    ds_features = layers.Reshape(int_shape)(ds_features)

    cur_x = ds_features

    for i in range(cnn_blocks):

        cur_x = layers.UpSampling2D((2, 2))(cur_x)

        cur_x = layers.Conv2D(cnn_depth*2**(cnn_blocks-i),

                              (3, 3),

                              padding='same',

                              activation='linear',

                              use_bias=False

                              )(cur_x)

        cur_x = layers.BatchNormalization()(cur_x)

        cur_x = layers.LeakyReLU(0.2)(cur_x)



    formed_output = layers.Conv2D(

        input_shape[2], (1, 1), activation='sigmoid')(cur_x)



    # instantiate decoder model

    decoder = models.Model(latent_inputs, formed_output, name='decoder')

    plot_model(decoder, to_file='vae_dcgan_decoder.png', show_shapes=True)



    # we have to reconfigure the model to instrument the output well

    # instantiate VAE model

    def rename_tensor(last_tensor, name): return layers.Lambda(

        lambda y: y, name=name)(last_tensor)

    enc_z_mean, enc_z_log_var, enc_z = [rename_tensor(c_lay, c_name) for c_name, c_lay in zip(

        ['enc_z_mean', 'enc_z_log_var', 'enc_z'], encoder(raw_inputs))]



    outputs = decoder(enc_z)

    vae = models.Model(inputs=[raw_inputs],

                       outputs=[outputs, enc_z_mean,

                                enc_z_log_var, enc_z],

                       name='vae_dcgan')

    vae.summary()



    vae.compile(optimizer='adam',

                loss={'enc_z_mean': z_mean_loss,

                      'enc_z_log_var': z_log_loss, 'decoder': recon_loss_func},

                loss_weights={'decoder': np.prod(

                    input_shape), 'enc_z_log_var': 0.5, 'enc_z_mean': 0.5},

                metrics={'decoder': 'mae'}

                )

    plot_model(vae, to_file='vae_dcgan.png', show_shapes=True)



    return encoder, decoder, vae
safe_shape = lambda x: np.stack(x.values, 0)[:, :, :, :]/255.0

def make_bundle(in_df):

    out_dict = {

            'decoder': safe_shape(in_df['image']),

            'enc_z_mean': np.zeros((in_df.shape[0], 2)),

            'enc_z_log_var': np.zeros((in_df.shape[0], 2)),     

        }

    return (

        {'encoder_input': safe_shape(in_df['image'])},

        out_dict

    )

train_bundle = make_bundle(train_df)

valid_bundle = make_bundle(test_df)

original_shape = train_bundle[0]['encoder_input'].shape[1:]

print(original_shape)
encoder, decoder, vae = build_dcgan_vae(original_shape, cnn_depth=16, cnn_blocks=5)

encoder.summary()
base_dcgan_vae_history = vae.fit(

    x=train_bundle[0],

    y=train_bundle[1],

    validation_data=valid_bundle,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE)

clear_output()
def show_training_results(**named_model_histories):

    model_out = list(named_model_histories.items())

    test_keys = [k for k in model_out[0]

                 [1].history.keys() if not k.startswith('val_')]

    fig, m_axs = plt.subplots(

        2, len(test_keys), figsize=(4*len(test_keys), 10))

    for c_key, (c_ax, val_ax) in zip(test_keys, m_axs.T):

        c_ax.set_title('Training: {}'.format(c_key.replace('_', ' ')))

        val_ax.set_title('Validation: {}'.format(c_key.replace('_', ' ')))

        for model_name, model_history in model_out:

            c_ax.plot(model_history.history[c_key], label=model_name)

            val_key = 'val_{}'.format(c_key)

            if val_key in model_history.history:

                val_ax.plot(

                    model_history.history[val_key], '-', label=model_name)



        c_ax.legend()

        val_ax.legend()





def plot_results(models,

                 data,

                 batch_size=128,

                 model_name="vae_mnist"):

    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments

        models (tuple): encoder and decoder models

        data (tuple): test data and label

        batch_size (int): prediction batch size

        model_name (string): which model is using this function

    """



    encoder, decoder = models

    x_test, y_vars = data

    os.makedirs(model_name, exist_ok=True)



    filename = os.path.join(model_name, "vae_mean.png")

    # display a 2D plot of the digit classes in the latent space

    z_mean, _, _ = encoder.predict(x_test,

                                   batch_size=batch_size)

    col_count = np.sqrt(len(y_vars)).astype(int)

    fig, m_ax = plt.subplots(col_count, col_count, figsize=(30, 30))

    for c_ax, (y_name, y_test) in zip(m_ax.flatten(), y_vars.items()):

        for k in np.unique(y_test):

            c_ax.plot(z_mean[y_test == k, 0], z_mean[y_test == k, 1],

                     '.', label='{:2.0f}'.format(k))

            c_ax.set_title(y_name)

        c_ax.legend()

        c_ax.set_xlabel("z[0]")

        c_ax.set_ylabel("z[1]")

    

    fig.savefig(filename)



    filename = os.path.join(model_name, "digits_over_latent.png")

    # display a 10x10 2D manifold of digits

    n = 10

    digit_size_x = original_shape[0]

    digit_size_y = original_shape[1]

    digit_shape_c = original_shape[2]

    figure = np.zeros((digit_size_x * n, digit_size_y * n, digit_shape_c))

    # linearly spaced coordinates corresponding to the 2D plot

    # of digit classes in the latent space

    grid_x = np.linspace(-4, 4, n)

    grid_y = np.linspace(-4, 4, n)[::-1]



    for i, yi in enumerate(grid_y):

        for j, xi in enumerate(grid_x):

            z_sample = np.array([[xi, yi]])

            x_decoded = decoder.predict(z_sample)

            digit = x_decoded[0].reshape(

                digit_size_x, digit_size_y, digit_shape_c)

            figure[i * digit_size_x: (i + 1) * digit_size_x,

                   j * digit_size_y: (j + 1) * digit_size_y] = digit



    plt.figure(figsize=(10, 10), dpi=300)

    plt.xlabel("z[0]")

    plt.ylabel("z[1]")

    plt.imshow(figure[:, :].squeeze(), cmap='Greys_r')

    plt.savefig(filename)

    imsave(filename+'_raw.png', figure)

    plt.show()



    features_test = encoder.predict(x_test)[-1]

    plt.hist2d(features_test[:, 0],

               features_test[:, 1],

               bins=30)



    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

    ax1.plot(features_test[:, 0])

    ax1.plot(features_test[:, 1])
show_training_results(

                      dcgan = base_dcgan_vae_history

                     )
plot_results((encoder, decoder),

             (safe_shape(test_df['image']), 

              {c_col: test_df[c_col] for c_col in val_count.keys()}),

             batch_size=BATCH_SIZE,

             model_name="vae_dcgan")