import os
try:
    # Create target Directory
    os.mkdir("/kaggle/working/recon_imgs0")
    os.mkdir("/kaggle/working/recon_imgs1")
    os.mkdir("/kaggle/working/generated_imgs0")
    os.mkdir("/kaggle/working/generated_imgs1")
    os.mkdir("/kaggle/working/model_checkpoints")
    os.mkdir("/kaggle/working/lin_int_imgs0")
    os.mkdir("/kaggle/working/lin_int_imgs1")
    os.mkdir("/kaggle/working/orginal_imgs")
except FileExistsError:
    print("Director already exists")

import shutil
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        #shutil.copy(os.path.join(dirname, filename),"/kaggle/working/model_checkpoints")
  


#!/usr/bin/env python3
import numpy as np

from keras import backend as K

from keras.layers import Layer
from functools import partial

def mean_gaussian_negative_log_likelihood(y_true, y_pred):
    nll = 0.5 *  K.square(y_pred - y_true)
#    nll = 0.5 * np.log(2 * np.pi) + 0.5 * K.square(y_pred - y_true)
    axis = tuple(range(1, len(K.int_shape(y_true))))
    return K.mean(K.sum(nll, axis=axis), axis=-1)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_pred, averaged_samples,
                          gradient_penalty_weight):
    #gradients = gradients[0]
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

class RandomWeightedAverage(Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:48:06 2020

@author: Noureldin Tawfek
"""

from tensorflow.keras.optimizers  import RMSprop,Adam

#
recon_depth = 7
seperate_di_batchs = False
include_zp_di = True
include_zp_di2 = False 
include_zp_de = True
include_zp_only_de = True
loss_fn = wasserstein_loss # 'binary_crossentropy' for vanilla gan, wasserstein_loss for wgan
recon_vs_gan_weight = 0
penalty_factor = 10
latent_dim=64
# training parameters
batch_size = 100
di_training_freq = 4
de_training_freq = 4
mini_batch = batch_size//di_training_freq
epochs = 10
dataset = "fashion_mnist"
#rmsprop = RMSprop(lr=0.0001)
config_initial_epoch = 0
optimizer = Adam(lr=0.0001, beta_1=0, beta_2=0.9)
optimizerEn = Adam(lr=0.0001, beta_1=0, beta_2=0.9, clipnorm=1.)

#!/usr/bin/env python3

#!/usr/bin/env python3


import os.path
import glob
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
from PIL import Image
import h5py
from numpy.random import choice
fileName ='/kaggle/input/celeba-cropped-faces/data_colab.h5'
from keras.datasets import mnist, fashion_mnist
#fileName = 'data_colab.h5'
from skimage import feature

if dataset=="nmist":
    (data_train, _), (data_test, _) = mnist.load_data()
elif dataset == "fashion_mnist":
    (data_train, _), (data_test, _) = fashion_mnist.load_data()


#NUM_SAMPLES = 202599
NUM_SAMPLES = data_train.shape[0]
#proj_root = os.path.split(os.path.dirname(__file__))[0]
#images_path = os.path.join(proj_root, 'img_align_celeba_png', '*.jpg')

data_train=data_train[0:NUM_SAMPLES]
data_test=data_test[0:300]


def rgb2gray(rgb):

    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def noisy_labels(y, p_flip):
	# determine the number of labels to flip
	n_select = int(p_flip * y.shape[0])
	# choose labels to flip
	flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
	# invert the labels in place
	y[flip_ix] = 1 - y[flip_ix]
	return y

def _load_image(f):
    im = Image.open(f) \
              .crop((0, 20, 178, 198)) \
              .resize((64, 64), Image.BICUBIC)
    return np.asarray(im)

def preproccess_data(data,normalize=True):
    x_train = data
    x_train_new = np.zeros((2,x_train.shape[0], 64, 64), dtype='int32')
    
    for i, img in enumerate(x_train):
        im = np.asarray(Image.fromarray(img).resize((64, 64), Image.BICUBIC))
        x_train_new[0,i] = im
        crop_filter = np.ones(im.shape).astype('u1')
        crop_filter[:,im.shape[1]//2 : im.shape[1]] = 0 # fill half the images with black pixels
        #im_2_modality = np.asarray(feature.canny(np.asarray(im),sigma=3).astype(int)*256)
        im_2_modality = im*crop_filter
        x_train_new[1,i] = np.asarray(im_2_modality)
        
    x_train = x_train_new.reshape(2,-1, 64, 64, 1)
    del x_train_new

    if normalize:
        x_train = x_train / 127.5 - 1.
        # To be sure
        x_train = np.clip(x_train, -1., 1.)
    
    return x_train

x_test = preproccess_data(data_test,normalize=True)

def celeba_loader(batch_size, normalize=True, num_child=4, seed=0, workers=8):
    rng = np.random.RandomState(seed)
#    images = glob.glob(images_path)
    while True:
#        rng.shuffle(images)
        with h5py.File(fileName, "r") as out:
            images = out['X_train']
            #images = rgb2gray(images[0:NUM_SAMPLES])[:,:,:,np.newaxis].astype('u1')
            images = images[0:NUM_SAMPLES]
            img_shape = images[0].shape
            crop_filter = np.ones(img_shape).astype('u1')
            crop_filter[:,img_shape[1]//2 : img_shape[1],:] = 0 # fill half the images with black pixels
            for s in range(0, len(images), batch_size):
                e = s + batch_size
                batch_images = images[s:e]
                batch_images = np.stack(batch_images)
                if normalize:
                    batch_images = batch_images / 127.5 - 1.
                    # To be sure
                    batch_images = np.clip(batch_images, -1., 1.)
                # Yield the same batch num_child times since the images will be consumed
                # by num_child different child generators
                for i in range(num_child):
                    yield batch_images,batch_images*crop_filter


def mnist_loader(batch_size, normalize=True, num_child=4, seed=0, workers=8):
    x_train = preproccess_data(data_train,normalize=True)

    #rng = np.random.RandomState(seed)
    while True:
        #rng.shuffle(x_train)
        for s in range(0, len(x_train), batch_size):
            e = s + batch_size
            batch_images = x_train[0,s:e]
            batch_images_2_modality = x_train[1,s:e]
            # Yield the same batch num_child times since the images will be consumed
            # by num_child different child generators
            for i in range(num_child):
                yield batch_images,batch_images_2_modality

                

def discriminator_loader(img_loader, latent_dim=latent_dim, seed=0):
    rng = np.random.RandomState(seed)
    while True:
        x1,x2 = next(img_loader)
        batch_size = x1.shape[0]
        # Sample z from isotropic Gaussian
        z_p = rng.normal(size=(batch_size, latent_dim))
        
        if loss_fn == wasserstein_loss:
            y_real = -1*np.ones((batch_size,), dtype='float32')
            y_fake = np.ones((batch_size,), dtype='float32')
            y_dummy = np.zeros((batch_size,), dtype='float32')
        else:
            y_real = np.ones((batch_size,), dtype='float32')
            y_fake = np.zeros((batch_size,), dtype='float32')


        yield [x1,x2,z_p],[y_real,y_real,y_fake,y_fake]

            

def decoder_loader(img_loader, latent_dim=latent_dim, seed=0):
    rng = np.random.RandomState(seed)
    factor = 1
    while True:
        x1,x2 = next(img_loader)
        batch_size = x1.shape[0]
        # Sample z from isotropic Gaussian
        z_p = rng.normal(size=(batch_size//factor, latent_dim))
        # Label as real
        if loss_fn == wasserstein_loss:
            y_real = -1*np.ones((batch_size//factor,), dtype='float32')
        else:
            y_real = np.ones((batch_size//factor,), dtype='float32')
        
        x1=x1[0:batch_size//factor]
        x2=x2[0:batch_size//factor]
        yield [x1,x2,z_p],[y_real,y_real]



def encoder_loader(img_loader):
    while True:
        x1,x2 = next(img_loader)
        yield [x1,x2], None
#!/usr/bin/env python3


import numpy as np

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense, Conv2DTranspose, Flatten, Reshape, \
    Lambda, LeakyReLU, Activation,Concatenate
from keras.regularizers import l2



from keras.optimizers import RMSprop

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
        
def set_trainable_multiple_models(trainable_models,untrainable_models):
    for model in trainable_models:
        set_trainable(model,True) 
    for model in untrainable_models:
        set_trainable(model,False) 

def create_models(n_channels=1, recon_depth=recon_depth, wdecay=1e-5, bn_mom=0.9, bn_eps=1e-6):

    image_shape = (64, 64, n_channels)
    n_encoder = 1024
    n_discriminator = 512
    decode_from_shape = (4, 4, 512)
    n_decoder = np.prod(decode_from_shape)
    leaky_relu_alpha = 0.2

    def conv_block(x, filters, leaky=True, transpose=False, name=''):
        conv = Conv2DTranspose if transpose else Conv2D
        activation = LeakyReLU(leaky_relu_alpha) if leaky else Activation('relu')
        layers = [
            conv(filters, 5, strides=2, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name=name + 'conv'),
            BatchNormalization(momentum=bn_mom, epsilon=bn_eps, name=name + 'bn'),
            activation
        ]
        if x is None:
            return layers
        for layer in layers:
            x = layer(x)
        return x

    # Encoder
    def create_encoder(model_name):
        x = Input(shape=image_shape, name='enc_input')

        y = conv_block(x, 64, name='enc_blk_1_')
        y = conv_block(y, 128, name='enc_blk_2_')
        y = conv_block(y, 256, name='enc_blk_3_')
        y = conv_block(y, 512, name='enc_blk_4_')
        y = Flatten()(y)
        y = Dense(n_encoder, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='enc_h_dense')(y)
        y = BatchNormalization(name='enc_h_bn')(y)
        y = LeakyReLU(leaky_relu_alpha)(y)

        z_mean = Dense(latent_dim, name='z_mean', kernel_initializer='he_uniform')(y)
        z_log_var = Dense(latent_dim, name='z_log_var', kernel_initializer='he_uniform')(y)

        return Model(x, [z_mean, z_log_var], name=model_name)

    # Decoder
    decoder = Sequential([
        Dense(n_decoder, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', input_shape=(latent_dim,), name='dec_h_dense'),
        BatchNormalization(name='dec_h_bn'),
        LeakyReLU(leaky_relu_alpha),
        Reshape(decode_from_shape),
        *conv_block(None, 512, transpose=True, name='dec_blk_1_'),
        *conv_block(None, 256, transpose=True, name='dec_blk_2_'),
        *conv_block(None, 128, transpose=True, name='dec_blk_3_'),
        *conv_block(None, 32, transpose=True, name='dec_blk_4_'),
        Conv2D(n_channels, 5, activation='tanh', padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dec_output')
    ], name='decoder1')
    
    decoder2 = Sequential([
        Dense(n_decoder, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', input_shape=(latent_dim,), name='dec_h_dense'),
        BatchNormalization(name='dec_h_bn'),
        LeakyReLU(leaky_relu_alpha),
        Reshape(decode_from_shape),
        *conv_block(None, 512, transpose=True, name='dec_blk_1_'),
        *conv_block(None, 256, transpose=True, name='dec_blk_2_'),
        *conv_block(None, 128, transpose=True, name='dec_blk_3_'),
        *conv_block(None, 32, transpose=True, name='dec_blk_4_'),
        Conv2D(n_channels, 5, activation='tanh', padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dec_output')
    ], name='decoder2')

    # Discriminator
    def create_discriminator(model_name):
        x = Input(shape=image_shape, name='dis_input')
        if loss_fn==wasserstein_loss:
            layers = [
                Conv2D(32, 5,strides=2,padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_blk_1_conv'),
                LeakyReLU(leaky_relu_alpha),
                Conv2D(128, 5, strides=2, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_blk_2_conv'),
                LeakyReLU(leaky_relu_alpha),
                Conv2D(256, 5, strides=2, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_blk_3_conv'),
                LeakyReLU(leaky_relu_alpha),
                Conv2D(256, 5, strides=2, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_blk_4_conv'),
                LeakyReLU(leaky_relu_alpha),
                Flatten(),
                Dense(n_discriminator, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_dense'),
                LeakyReLU(leaky_relu_alpha),
                Dense(1, activation=None, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_output')]
        else :
            layers = [
                Conv2D(32, 5, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_blk_1_conv'),
                LeakyReLU(leaky_relu_alpha),
                *conv_block(None, 128, leaky=True, name='dis_blk_2_'),
                *conv_block(None, 256, leaky=True, name='dis_blk_3_'),
                *conv_block(None, 256, leaky=True, name='dis_blk_4_'),
                Flatten(),
                Dense(n_discriminator, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_dense'),
                BatchNormalization(name='dis_bn'),
                LeakyReLU(leaky_relu_alpha),
                Dense(1, activation='sigmoid', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_output')
            ]

        y = x
        y_feat = None
        for i, layer in enumerate(layers, 1):
            y = layer(y)
            # Output the features at the specified depth
            if i == recon_depth:
                y_feat = y

        return Model(x, [y, y_feat], name=model_name)

    encoders = [create_encoder(model_name="encoder"+str(i+1)) for i in range(2)]
    discriminators = [ create_discriminator(model_name="discriminator"+str(i+1)) for i in range(2) ]
    decoders = [decoder,decoder2]

    return encoders, decoders, discriminators


def _sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
       Instead of sampling from Q(z|X), sample eps = N(0,I)

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

def _averageweighting(args):
    real_imgs,fake_imgs=args
    batch_size = K.shape(real_imgs)[0]
    alpha = K.random_uniform(shape=(batch_size, 1, 1, 1))
    return (alpha * real_imgs) + ((1 - alpha) * fake_imgs)

def _product_of_experts(args):
    mu1,logvar1,mu2,logvar2 = args
    T1         = 1. / K.exp(logvar1) 
    T2         = 1. / K.exp(logvar2)
    pd_mu     = (mu1*T1+mu2*T2) / (T1+T2)
    pd_var    = 1. / (T1+T2)
    pd_logvar = K.log(pd_var)
    return [pd_mu, pd_logvar]

def _calculate_gradient_penalty(args,discriminator):
    image_shape=K.int_shape(args[0])[1:]
    weighter = Lambda(_averageweighting,output_shape=image_shape, name='weighter')
    # average image
    epsilon = K.random_uniform(shape=(image_shape[0], 1, 1, 1))
    a_img = weighter(args)
    
    # gradient penalty  <this is point of WGAN-gp>
    a_out=discriminator(a_img)[0]
    gradients = K.gradients(a_out, a_img)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    penalty =  K.mean(penalty_factor*K.square(1 - gradient_l2_norm))
    return penalty



def build_graph(encoders, decoders, discriminators, recon_vs_gan_weight=recon_vs_gan_weight):
    image_shape = K.int_shape(encoders[0].input)[1:]
    latent_shape = K.int_shape(decoders[0].input)[1:]
    
    sampler = Lambda(_sampling, output_shape=latent_shape, name='sampler')
    
    poe = Lambda(_product_of_experts,name='poe')
    #gp_calculator =Lambda(_calculate_gradient_penalty,output_shape=latent_shape,name='gp_calculator')
    # Inputs
    x1 = Input(shape=image_shape, name='input_image1')
    x2 = Input(shape=image_shape, name='input_image2')
    kl_loss_lamda = Input(shape=(1,), name="kl_loss_lamda" )
    x = [x1,x2]
    # z_p is sampled directly from isotropic gaussian
    z_p = Input(shape=latent_shape, name='z_p')

    # Build computational graph
     
    mu1, logvar1 = encoders[0](x[0])
    mu2, logvar2 = encoders[1](x[1])
    z_mean, z_log_var = poe([mu1,logvar1,mu2,logvar2])
    
    z = sampler([z_mean, z_log_var])
    z1 = sampler([ mu1, logvar1])
    z2 = sampler([ mu2, logvar2])
    
    x_tilde = [ decoder(z) for decoder in decoders ] 
    x_tilde_z1 = [ decoder(z1) for decoder in decoders ] 
    x_tilde_z2 = [ decoder(z2) for decoder in decoders ] 
    
    x_p = [decoder(z_p) for decoder in decoders]
    
    
    dis_x_outs = [ discriminator(xi) for discriminator,xi in zip(discriminators, x) ]
    
    dis_x_tilde_outs = [ discriminator(xi) for discriminator,xi in zip(discriminators, x_tilde) ]
    dis_x_tilde_outs_z1 = [ discriminator(xi) for discriminator,xi in zip(discriminators, x_tilde_z1) ]
    dis_x_tilde_outs_z2 = [ discriminator(xi) for discriminator,xi in zip(discriminators, x_tilde_z2) ]
    
    dis_x_p = [ discriminator(xi)[0] for discriminator,xi in zip(discriminators, x_p) ]

    
    penalty = [ _calculate_gradient_penalty([x[i],x_p[i]],discriminators[i]) for i in range(len(x))]

    # Learned similarity metric : dis_x_outs[i][j] feature map of input x[j]
    dis_nll_loss = [mean_gaussian_negative_log_likelihood(dis_x_outs[i][1], dis_x_tilde_outs[i][1]) for i in range(2) ]
    dis_nll_loss1 = [mean_gaussian_negative_log_likelihood(dis_x_outs[i][1], dis_x_tilde_outs_z1[i][1]) for i in range(2) ]
    dis_nll_loss2 = [mean_gaussian_negative_log_likelihood(dis_x_outs[i][1], dis_x_tilde_outs_z2[i][1]) for i in range(2) ]
    # KL divergence loss
    kl_loss = 10*K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
    kl_loss1 = 10*K.mean(-0.5 * K.sum(1 + logvar1 - K.square(mu1) - K.exp(logvar1), axis=-1))
    kl_loss2 = 10*K.mean(-0.5 * K.sum(1 + logvar2 - K.square(mu2) - K.exp(logvar2), axis=-1))
    
    # Create models for training
    
    encoder_train = Model([x[0],x[1],kl_loss_lamda], [dis_x_tilde_outs[0][1],dis_x_tilde_outs_z1[0][1],dis_x_tilde_outs_z2[0][1]], name='e')
    
    encoder_train.add_loss(kl_loss) 
    encoder_train.add_loss(kl_loss1) 
    encoder_train.add_loss(kl_loss2) 
    
    [ encoder_train.add_loss(loss) for loss in dis_nll_loss ]
    [ encoder_train.add_loss(loss) for loss in dis_nll_loss1 ]
    [ encoder_train.add_loss(loss) for loss in dis_nll_loss2 ]
    
    
    
    decoder_train = Model([x[0],x[1],z_p],[dis_x_p[0],dis_x_p[1]], name='de')
    #decoder_train = Model([x[0],x[1],z_p],[dis_x_tilde_outs[0][0],dis_x_tilde_outs[1][0]], name='de')

    discriminator_train = Model([x[0],x[1],z_p] ,[dis_x_outs[0][0],dis_x_outs[1][0],dis_x_p[0],dis_x_p[1]], name='di')   
    #discriminator_train = Model([x[0],x[1],z_p] ,[dis_x_outs[0][0],dis_x_outs[1][0],dis_x_tilde_outs[0][0],dis_x_tilde_outs[1][0]], name='di') 
    [ discriminator_train.add_loss(loss) for loss in penalty ]
    
    
    # Additional models for testing
    vae = Model([x[0],x[1]], [x_tilde[0],x_tilde[1]], name='vae')
    vaegan = Model([x[0],x[1]], [dis_x_tilde_outs[0][0],dis_x_tilde_outs[1][0]], name='vaegan')
    
    
    # compiling models
    set_trainable_multiple_models(encoders,discriminators + decoders )
    encoder_train.compile(optimizerEn)
    
    
    set_trainable_multiple_models(discriminators,decoders + encoders)
    discriminator_train.compile(optimizer, loss=[loss_fn]*4)
    
    set_trainable_multiple_models(decoders,discriminators + encoders)
    decoder_train.compile(optimizer, loss=[loss_fn]*2)
    
    encoder_train.summary()
    decoder_train.summary()
    discriminator_train.summary()
   
    # add custom loss to metrics
    encoder_train.add_metric(kl_loss, name= "kl_loss") 
    encoder_train.add_metric(kl_loss1, name= "kl_loss1") 
    encoder_train.add_metric(kl_loss2, name= "kl_loss2") 
    
    
    [ encoder_train.add_metric(dis_nll_loss,name ="dis_nll_loss_joint"+str(i)) for i in range(len(dis_nll_loss)) ]
    [ encoder_train.add_metric(dis_nll_loss1,name ="dis_nll_loss_1m"+str(i)) for i in range(len(dis_nll_loss1)) ]
    [ encoder_train.add_metric(dis_nll_loss2,name ="dis_nll_loss_2m"+str(i)) for i in range(len(dis_nll_loss2)) ]
    
    [ discriminator_train.add_metric(penalty , name = "partial_gp_loss1") for pen in penalty ]
    
    
    
    
    vaegan.summary()
    
    print(discriminators[0].layers[recon_depth].name)
    
    set_trainable(vaegan, True)
    return encoder_train, decoder_train, discriminator_train, vae, vaegan
#    return encoder_train, decoder_train, discriminator_train, vae, vaegan,kl_loss,dis_nll_loss,normal_dis_nll_loss





from keras import callbacks as cbks
import numpy as np
from PIL import Image


def fit_models(callback_model,
               models,
               generators,
               batch_size,
               steps_per_epoch=None,
               epochs=1,
               verbose=0,
               callbacks=None,
               initial_epoch=0):
    epoch = initial_epoch

    # Prepare display labels.
    callback_metrics = sum([modelt.metrics_names for modelt in models],[])

    # prepare callbacks
    stateful_metric_names = []
    for model in models:
       model.history = cbks.History()
#         try:
#             stateful_metric_names.extend(model.stateful_metric_names)
#         except AttributeError:
#             stateful_metric_names.extend(model.model.stateful_metric_names)
# =============================================================================
    _callbacks = [cbks.BaseLogger(stateful_metrics=None)]
    if verbose:
         _callbacks.append(
            cbks.ProgbarLogger(
                count_mode='steps',
                stateful_metrics=None))
    _callbacks += (callbacks or []) + [model.history for model in models]
    
    callbacks = _callbacks
    
    [callback.set_model(callback_model) for callback in callbacks ]
    [ callback.set_params({
         'epochs': epochs,
         'steps': steps_per_epoch,
         'verbose': verbose,
         'do_validation': False,
         'metrics': callback_metrics,
     }) for callback in callbacks ]
# =============================================================================
    
    [callback.on_train_begin() for callback in callbacks]
    try:
        callback_model.stop_training = False
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
# =============================================================================
#             for model in models:
#                 try:
#                     stateful_metric_functions = model.stateful_metric_functions
#                 except AttributeError:
#                     stateful_metric_functions = model.model.stateful_metric_functions
#                 for m in stateful_metric_functions:
#                     m.reset_states()
# =============================================================================
            [callback.on_epoch_begin(epoch) for callback in callbacks]
            steps_done = 0
            batch_index = 0
            
            while steps_done < steps_per_epoch:
                print("epoche: "+ str(epoch) + " batch : "+ str(batch_index))
                # build batch logs
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                [callback.on_batch_begin(batch_index, batch_logs) for callback in callbacks]

                for model, output_generator in zip(models, generators):
                   
                    metrics= model.metrics_names
                    generator_output = next(output_generator)
                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))

                    if len(generator_output) == 2:
                        x, y = generator_output
                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
# =============================================================================
                    if model.name=="di":
                        
                        for i in range(di_training_freq):
                            x_minibatch=[xi[mini_batch*i:mini_batch*(i+1)] for xi in x]
                            y_minibatch=[yi[mini_batch*i:mini_batch*(i+1)] for yi in y]
                            outs = model.train_on_batch(x_minibatch, y_minibatch, sample_weight=sample_weight)
                            print(model.name)
                            print(model.metrics_names)
                            print(outs)
                    elif model.name=="de":
                        for i in range(de_training_freq):
                            x_minibatch=[xi[mini_batch*i:mini_batch*(i+1)] for xi in x]
                            y_minibatch=[yi[mini_batch*i:mini_batch*(i+1)] for yi in y]
                            outs = model.train_on_batch(x_minibatch, y_minibatch, sample_weight=sample_weight)
                            print(model.name)
                            print(model.metrics_names)
                            print(outs)
                    else :
                        kl_loss_lamda = min(epoch/40,1)*np.ones((batch_size,1)) # linearly annealing kl loss lamda 
                        x.append(kl_loss_lamda)
                        outs = model.train_on_batch(x, y, sample_weight=sample_weight)
                        print(model.name)
                        print(model.metrics_names)
                        print(outs)
                    if not isinstance(outs, list):
                        outs = [outs]
                    for i, name in enumerate(metrics):
                        batch_logs[name] = outs[i]
                    
                    batch_logs["data_sample"] = x_test[0:16]
                    
                [callback.on_batch_end(batch_index, batch_logs) for callback in callbacks]
                
                batch_index += 1
                steps_done += 1

                # Epoch finished.
                if callback_model.stop_training:
                    break

            [callback.on_epoch_end(epoch, epoch_logs) for callback in callbacks]
            
            epoch += 1
            if callback_model.stop_training:
                break

    finally:
        pass

    [callback.on_train_end() for callback in callbacks]

    return [model.history for model in models]


#!/usr/bin/env python3


from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from keras.callbacks import Callback

#generated_path = ""
generated_path = "/kaggle/working/"
class DecoderSnapshot(Callback):

    def __init__(self, step_size=200, latent_dim=latent_dim, decoder_index=-2):
        super().__init__()
        self._step_size = step_size
        self._steps = 0
        self._epoch = 0
        self._latent_dim = latent_dim
        self._decoder_index = decoder_index
        self._img_rows = 64
        self._img_cols = 64
        self._thread_pool = ThreadPoolExecutor(1)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._steps = 0

    def on_batch_begin(self, batch, logs=None):
        self._steps += 1
    
    def on_batch_end(self, batch, logs=None):
        print((self._steps-1) % self._step_size )
        if (self._steps-1) % 100 == 0:
            self.plot_images(logs=logs)


    def plot_images(self, samples=16,logs=None):
        decoder1 = self.model.get_layer('decoder1')
        decoder2 = self.model.get_layer('decoder2')
        encoder1 = self.model.get_layer('encoder1')
        encoder2 = self.model.get_layer('encoder2')
        sampler = self.model.get_layer('sampler')
        
        # create generated images from normal distribution N(0,1)
        z = np.random.normal(size=(samples, self._latent_dim))
        for i,decoder in enumerate([decoder1,decoder2]):
            generated_images = decoder.predict(z)
            filename = 'generated_imgs'+str(i)+'/modality'+str(i)+'generated_%d_%d.png' % (self._epoch, self._steps)
            self.save_plot(generated_images, filename)

        data=logs["data_sample"]
        self.save_plot(data[0][0:16], 'orginal_imgs/modality0orginal_%d_%d.png'% (self._epoch, self._steps))
        self.save_plot(data[1][0:16], 'orginal_imgs/modality1orginal_%d_%d.png'% (self._epoch, self._steps))
        mu1, logvar1 = encoder1.predict(data[0][0:16])
        mu2, logvar2 = encoder2.predict(data[1][0:16])
        z_parameters = _product_of_experts([mu1,logvar1,mu2,logvar2]) 
        z=_sampling(z_parameters)
        z1 = _sampling([mu1, logvar1])
        z2 = _sampling([mu2, logvar2])
        
        # reconstruct im
        for j,zi in enumerate([z,z1,z2]):
            for i,decoder in enumerate([decoder1,decoder2]):
                recon_imgs = decoder.predict(zi,steps=1)
                filename = 'recon_imgs'+str(i)+'/modality'+str(i)+'recon_z'+str(j)+'_%d_%d.png' % (self._epoch, self._steps)
                self.save_plot(recon_imgs, filename)
            
        # create generated images from normal distribution N(0,1)
        n = 2  # for 2 random indices
        index = np.random.choice(z.shape[0], n, replace=False) 
        z0=z[index[0]]
        z1=z[index[1]]
        ratio= (np.arange(16)/15) 
        z=z0+(z1-z0)*ratio[:,None] # linear interpolation over two vectors in latent space
        for i,decoder in enumerate([decoder1,decoder2]):
            print(z[1,0:5])
            print(z[15,0:5])
            lin_int_imgs = decoder.predict(z,steps=1)
            filename = 'lin_int_imgs'+str(i)+'/modality'+str(i)+'lin_int_imgs_%d_%d.png' % (self._epoch, self._steps)
            self.save_plot(lin_int_imgs, filename)
        
        
    @staticmethod
    def save_plot(images, filename):
        images = (images + 1.) * 127.5
        images = np.clip(images, 0., 255.)
        images = images.astype('uint8')
        rows = []
        for i in range(0, len(images), 4):
            rows.append(np.concatenate(images[i:(i + 4), :, :, :], axis=0))
        plot = np.concatenate(rows, axis=1).squeeze()
        
        Image.fromarray(plot).save(filename)


class ModelsCheckpoint(Callback):

    def __init__(self, epoch_format, *models):
        super().__init__()
        self._epoch_format = epoch_format
        self._models = models

    def on_epoch_end(self, epoch, logs=None):
        suffix = self._epoch_format.format(epoch=epoch + 1, **logs)
        if epoch % 10 == 0:
            for model in self._models:
                model.save_weights('model_checkpoints/'+model.name + suffix)


#!/usr/bin/env python3

import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import TensorBoard

from keras.losses import mean_absolute_error

model_path = ""
#model_path = "/kaggle/working/model_checkpoints/"


def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def main():
    encoders, decoders, discriminators = create_models(n_channels=1)
    encoder_train, decoder_train, discriminator_train,vae, vaegan= build_graph(encoders, decoders, discriminators)
    #encoder_train, decoder_train, discriminator_train,vae, vaegan , kl_loss,dis_nll_loss,normal_dis_nll_loss = build_graph(encoder, decoder, discriminator)

    try:
        initial_epoch = int(sys.argv[1])
    except (IndexError, ValueError):
        initial_epoch = config_initial_epoch

    epoch_format = '.{epoch:03d}.h5'

    if initial_epoch != 0:
        suffix = epoch_format.format(epoch=initial_epoch)
        encoders[0].load_weights(model_path+'encoder' + suffix)
        encoders[1].load_weights(model_path+'encoder' + suffix)
        decoders[0].load_weights(model_path+'decoder' + suffix)
        decoders[1].load_weights(model_path+'decoder' + suffix)
        discriminators[0].load_weights(model_path+'discriminator' + suffix)
        discriminators[1].load_weights(model_path+'discriminator' + suffix)



    
    

    checkpoint = ModelsCheckpoint(epoch_format, encoders[0],encoders[1], decoders[0],decoders[1], discriminators[0],discriminators[1])
    decoder_sampler = DecoderSnapshot()

    callbacks = [checkpoint, decoder_sampler]

    

    steps_per_epoch = NUM_SAMPLES // batch_size

    seed = np.random.randint(0,2**32 - 1,None,"int64")
    num_child = 3
    #img_loader = celeba_loader(batch_size, num_child=num_child, seed=seed)
    img_loader = mnist_loader(batch_size, num_child=num_child, seed=seed)

    dis_loader = discriminator_loader(img_loader, seed=seed)
    dec_loader = decoder_loader(img_loader, seed=seed)
    enc_loader = encoder_loader(img_loader)
 
    models = [discriminator_train, decoder_train, encoder_train]
    generators = [dis_loader,dec_loader, enc_loader]

    
#    metrics = [{'di_l': 1, 'di_l_t': 2, 'di_l_p': 3, 'di_a': 4, 'di_a_t': 7, 'di_a_p': 10}, {'de_l_t': 1, 'de_l_p': 2, 'de_a_t': 3, 'de_a_p': 5}, {'en_l': 0}]

    histories = fit_models(vaegan, models, generators, batch_size,
                           steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                           epochs=epochs, initial_epoch=initial_epoch)

    with open('histories.pickle', 'wb') as f:
        pickle.dump(histories, f)



if __name__ == '__main__':
    main()
