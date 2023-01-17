# example of pix2pix gan for satellite to map image-to-image translation

from numpy import load

from numpy import zeros

from numpy import ones

from numpy.random import randint

from keras.optimizers import Adam

from keras.initializers import RandomNormal

from keras.models import Model

from keras.models import Input

from keras.layers import Conv2D

from keras.layers import Conv2DTranspose

from keras.layers import LeakyReLU

from keras.layers import Activation

from keras.layers import Concatenate

from keras.layers import Dropout

from keras.layers import BatchNormalization

from keras.layers import LeakyReLU

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate

from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model

from keras.optimizers import Adam

from matplotlib import pyplot

import sys

from glob import glob

import cv2

import numpy as np
# define the discriminator model

def define_discriminator(image_shape):

    # weight initialization

    init = RandomNormal(stddev=0.02)

    # source image input

    in_src_image = Input(shape=image_shape)

    # target image input

    in_target_image = Input(shape=image_shape)

    # concatenate images channel-wise

    merged = Concatenate()([in_src_image, in_target_image])

    # C64

    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)

    d = LeakyReLU(alpha=0.2)(d)

    # C128

    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)

    d = BatchNormalization()(d)

    d = LeakyReLU(alpha=0.2)(d)

    # C256

    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)

    d = BatchNormalization()(d)

    d = LeakyReLU(alpha=0.2)(d)

    # C512

    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)

    d = BatchNormalization()(d)

    d = LeakyReLU(alpha=0.2)(d)

    # second last output layer

    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)

    d = BatchNormalization()(d)

    d = LeakyReLU(alpha=0.2)(d)

    # patch output

    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)

    patch_out = Activation('sigmoid')(d)

    # define model

    model = Model([in_src_image, in_target_image], patch_out)

    # compile model

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

    return model
def conv2d(layer_input, filters, f_size=4, bn=True):

    """Layers used during downsampling"""

    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)

    d = LeakyReLU(alpha=0.2)(d)

    if bn:

        d = BatchNormalization(momentum=0.8)(d)

    return d



def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):

    """Layers used during upsampling"""

    u = UpSampling2D(size=2)(layer_input)

    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)

    if dropout_rate:

        u = Dropout(dropout_rate)(u)

    u = BatchNormalization(momentum=0.8)(u)

    u = Concatenate()([u, skip_input])

    return u



# define the standalone generator model

def define_generator(image_shape=(256, 256, 3)):

    """U-Net Generator"""

    gf=64

    

    # Image input

    d0 = Input(shape=image_shape)



    # Downsampling

    d1 = conv2d(d0, gf, bn=False)

    d2 = conv2d(d1, gf*2)

    d3 = conv2d(d2, gf*4)

    d4 = conv2d(d3, gf*8)

    d5 = conv2d(d4, gf*8)

    d6 = conv2d(d5, gf*8)

    d7 = conv2d(d6, gf*8)



    # Upsampling

    u1 = deconv2d(d7, d6, gf*8)

    u2 = deconv2d(u1, d5, gf*8)

    u3 = deconv2d(u2, d4, gf*8)

    u4 = deconv2d(u3, d3, gf*4)

    u5 = deconv2d(u4, d2, gf*2)

    u6 = deconv2d(u5, d1, gf)



    u7 = UpSampling2D(size=2)(u6)

    output_img = Conv2D(image_shape[2], kernel_size=4, strides=1, padding='same', activation='tanh')(u7)



    return Model(d0, output_img)



# define the combined generator and discriminator model, for updating the generator

def define_gan(g_model, d_model, image_shape):

    # make weights in the discriminator not trainable

    d_model.trainable = False

    # define the source image

    in_src = Input(shape=image_shape)

    # connect the source image to the generator input

    gen_out = g_model(in_src)

    # connect the source input and generator output to the discriminator input

    dis_out = d_model([in_src, gen_out])

    # src image as input, generated image and classification output

    model = Model(in_src, [dis_out, gen_out])

    # compile model

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])

    return model
# load and prepare training images

def load_real_samples(filename):

    # load compressed arrays

    data = load(filename)

    # unpack arrays

    X1, X2 = data['arr_0'], data['arr_1']

    # scale from [0,255] to [-1,1]

    X1 = (X1 - 127.5) / 127.5

    X2 = (X2 - 127.5) / 127.5

    return [X1, X2]



# select a batch of random samples, returns images and target

def generate_real_samples(dataset, n_samples, patch_shape):

    # choose random instances

    ix = randint(0, len(dataset), n_samples)

    # retrieve selected images

    X1 = np.zeros((n_samples,512,512,3))

    X2 = np.zeros((n_samples,512,512,3))

    cnt = 0

    for i in ix:

        img = cv2.imread(dataset[i])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        X1_img = img[:,:512,:]

        X2_img = img[:,512:,:]

        X1[cnt] = X1_img

        X2[cnt] = X2_img

        cnt+=1

    X1 = (X1 - 127.5) / 127.5

    X2 = (X2 - 127.5) / 127.5

    # generate 'real' class labels (1)

    y = ones((n_samples, patch_shape, patch_shape, 1))

    return [X1, X2], y



# generate a batch of images, returns images and targets

def generate_fake_samples(g_model, samples, patch_shape):

    # generate fake instance

    X = g_model.predict(samples)

    # create 'fake' class labels (0)

    y = zeros((len(X), patch_shape, patch_shape, 1))

    return X, y



# generate samples and save as a plot and save the model

def summarize_performance(step, g_model, dataset, n_samples=3):

    # select a sample of input images

    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)

    # generate a batch of fake samples

    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

    # scale all pixels from [-1,1] to [0,1]

    X_realA = (X_realA + 1) / 2.0

    X_realB = (X_realB + 1) / 2.0

    X_fakeB = (X_fakeB + 1) / 2.0

    # plot real source images

    pyplot.figure(figsize=(30,30), constrained_layout=True)

    for i in range(n_samples):

        pyplot.subplot(3, n_samples, 1 + i)

        pyplot.axis('off')

        pyplot.imshow(X_realA[i])

    # plot generated target image

    for i in range(n_samples):

        pyplot.subplot(3, n_samples, 1 + n_samples + i)

        pyplot.axis('off')

        pyplot.imshow(X_fakeB[i])

    # plot real target image

    for i in range(n_samples):

        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)

        pyplot.axis('off')

        pyplot.imshow(X_realB[i])

    # save plot to file

    filename1 = 'plot_%d.png' % (step+1)

    pyplot.savefig(filename1)

    pyplot.close()

    # save the generator model

    filename2 = 'model_%d.h5' % (step+1)

    g_model.save(filename2)

    print('>Saved: %s and %s' % (filename1, filename2))
# train pix2pix models

from tqdm import tqdm

def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):

    # determine the output square shape of the discriminator

    n_patch = d_model.output_shape[1]

    # calculate the number of batches per training epoch

    bat_per_epo = int(len(dataset) / n_batch)

    # calculate the number of training iterations

    n_steps = bat_per_epo * n_epochs

    # manually enumerate epochs

    t = tqdm(range(n_steps), desc='ML')

    for i in t:

        # select a batch of real samples

        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)

        # generate a batch of fake samples

        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        # update discriminator for real samples

        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)

        # update discriminator for generated samples

        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        # update the generator

        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

        # summarize performance

        #sys.stderr.write('\r')

        # the exact output you're looking for:

        t.set_description('>%d, d1[%.6f] d2[%.6f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

        #print(('>%d, d1[%.6f] d2[%.6f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss)))

        #sys.stderr.flush()

        # summarize model performance

        if (i+1) % (bat_per_epo*5) == 0:

            summarize_performance(i/bat_per_epo, g_model, dataset)
# load image data

dataset = glob('/kaggle/input/one-piece/Dataset/*')

#dataset = load_real_samples('maps_256.npz')

#dataset = [src_images, tar_images]

#print('Loaded', dataset[0].shape, dataset[1].shape)

len(dataset)
from keras.models import load_model

# define input shape based on the loaded dataset

image_shape = (512,512,3)

# define the models

d_model = define_discriminator(image_shape)

#g_model = define_generator(image_shape)

g_model = load_model('/kaggle/input/model-unet/model_7.h5', compile=False)

# define the composite model

gan_model = define_gan(g_model, d_model, image_shape)

# train model+++++++++++++++++++++
train(d_model, g_model, gan_model, dataset, n_epochs=25)