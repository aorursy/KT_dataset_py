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
except FileExistsError:
    print("Director already exists")
    
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))    
import os
for i in range(23):
    try:
        os.remove("/kaggle/working/model_checkpoints/decoder1."+ "{0:0=3d}".format(i+1)+".h5")
    except OSError:
        pass    
for i in range(23):
    try:
        os.remove("/kaggle/working/model_checkpoints/decoder2."+ "{0:0=3d}".format(i+1)+".h5")
    except OSError:
        pass
for i in range(24):
    try:
        os.remove("/kaggle/working/model_checkpoints/encoder1."+ "{0:0=3d}".format(i+1)+".h5")
    except OSError:
        pass
for i in range(24):
    try:
        os.remove("/kaggle/working/model_checkpoints/encoder2."+ "{0:0=3d}".format(i+1)+".h5")
    except OSError:
        pass

#!/usr/bin/env python3
import numpy as np

from keras import backend as K

from keras.layers import Layer
from functools import partial
from keras.losses import MeanSquaredError as mse

def mean_gaussian_negative_log_likelihood(y_true, y_pred):
    nll = 0.5 *  K.square(y_pred - y_true)
#    nll = 0.5 * np.log(2 * np.pi) + 0.5 * K.square(y_pred - y_true)
    axis = tuple(range(1, len(K.int_shape(y_true))))
    return K.mean(K.sum(nll, axis=axis), axis=-1)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def scaled_mse(y_true, y_pred):
    return 64*K.mean(K.square(y_pred - y_true), axis=-1)

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
#!/usr/bin/env python3ureldin Tawfek


from tensorflow.keras.optimizers  import RMSprop,Adam

#
recon_depth = 7
seperate_di_batchs = False
include_zp_di = True
include_zp_di2 = False 
include_zp_de = True
include_zp_only_de = True
loss_fn = scaled_mse #wasserstein_loss # 'binary_crossentropy' for vanilla gan, wasserstein_loss for wgan
recon_vs_gan_weight = 0
penalty_factor = 10
latent_dim=64
# training parameters
batch_size = 100
mini_batch = batch_size
epochs = 30
dataset = "fashion_mnist"
#rmsprop = RMSprop(lr=0.0001)
config_initial_epoch = 0
optimizer = Adam(lr=0.0001, beta_1=0, beta_2=0.9)
optimizerEn = Adam(lr=0.0001, beta_1=0, beta_2=0.9, clipnorm=1.)



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
NUM_SAMPLES = 10000 # data_train.shape[0]
#proj_root = os.path.split(os.path.dirname(__file__))[0]
#images_path = os.path.join(proj_root, 'celebacropped', '*.jpg')

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

# =============================================================================
# def celeba_loader(batch_size, normalize=True, num_child=4, seed=0, workers=8):
#     rng = np.random.RandomState(seed)
#     images = glob.glob(images_path)
#     while True:
#         rng.shuffle(images)
#         with h5py.File(fileName, "r") as out:
#             images = out['X_train']
#             #images = rgb2gray(images[0:NUM_SAMPLES])[:,:,:,np.newaxis].astype('u1')
#             images = images[0:NUM_SAMPLES]
#             img_shape = images[0].shape
#             crop_filter = np.ones(img_shape).astype('u1')
#             crop_filter[:,img_shape[1]//2 : img_shape[1],:] = 0 # fill half the images with black pixels
#             for s in range(0, len(images), batch_size):
#                 e = s + batch_size
#                 batch_images = images[s:e]
#                 batch_images = np.stack(batch_images)
#                 if normalize:
#                     batch_images = batch_images / 127.5 - 1.
#                     # To be sure
#                     batch_images = np.clip(batch_images, -1., 1.)
#                 # Yield the same batch num_child times since the images will be consumed
#                 # by num_child different child generators
#                 for i in range(num_child):
#                     yield batch_images,batch_images*crop_filter
# =============================================================================
def celeba_loader(batch_size, normalize=True, num_child=4, seed=0, workers=8):
   
    rng = np.random.RandomState(seed)
    images = glob.glob(images_path)
    with Pool(workers) as p:
        while True:
            
            rng.shuffle(images)
            crop_filter = np.ones(shape=(64,64,3)).astype('u1')
            crop_filter[:,64//2 : 64,:] = 0 # fill half the images with black pixels
            for s in range(0, len(images), batch_size):
                e = s + batch_size
                batch_names = images[s:e]
                batch_images = p.map(_load_image, batch_names)
                batch_images = np.stack(batch_images)
                if s == 0:
                    Image.fromarray(batch_images[0]).show()
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

                

def encoder_loader(img_loader):
    while True:
        x1,x2 = next(img_loader)
        yield [x1,x2], [x1,x2,x1,x2,x1,x2]
#!/usr/bin/env python3


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

            
    encoders = [create_encoder(model_name="encoder"+str(i+1)) for i in range(2)]
    decoders = [decoder,decoder2]

    return encoders, decoders



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


def _product_of_experts(args):
    
    mu=args[:len(args)//2]
    logvar = args[len(args)//2:]
    T         = 1. / K.exp(logvar)
    print("T.shape")
    print(T.shape)
    pd_mu     = (K.sum(mu*T, axis = 0) + 0) / (K.sum(T,axis = 0)+1) # mu_prior*T_prior = 0 , T_prior=1
    pd_var    = 1. / (K.sum(T,axis = 0)+1)
    pd_logvar = K.log(pd_var)
    return [pd_mu, pd_logvar]


def build_graph(encoders, decoders):
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
    

    # Build computational graph
     
    mu1, logvar1 = encoders[0](x[0])
    mu2, logvar2 = encoders[1](x[1])
    
    z_mean, z_log_var = poe([mu1,mu2,logvar1,logvar2])
    z1_mean, z1_log_var = poe([mu1,logvar1])
    z2_mean, z2_log_var = poe([mu2,logvar2])

    
    z = sampler([z_mean, z_log_var])
    z1 = sampler([ z1_mean, z1_log_var])
    z2 = sampler([ z2_mean, z2_log_var])
    
    x_tilde = [ decoder(z) for decoder in decoders ] 
    x_tilde_z1 = [ decoder(z1) for decoder in decoders ] 
    x_tilde_z2 = [ decoder(z2) for decoder in decoders ] 
    
    
    
    # KL divergence loss
    print(z_mean.shape)
    kl_loss = kl_loss_lamda*K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
    kl_loss1 =kl_loss_lamda*K.mean(-0.5 * K.sum(1 + z1_log_var - K.square(z1_mean) - K.exp(z1_log_var), axis=-1))
    kl_loss2 = kl_loss_lamda*K.mean(-0.5 * K.sum(1 + z2_log_var - K.square(z2_mean) - K.exp(z2_log_var), axis=-1))
    
    encoder_train = Model([x[0],x[1],kl_loss_lamda], x_tilde+x_tilde_z1+x_tilde_z2, name='e')
    
    encoder_train.add_loss(kl_loss) 
    encoder_train.add_loss(kl_loss1) 
    encoder_train.add_loss(kl_loss2) 
    
    

    # compiling models
    set_trainable_multiple_models(encoders+ decoders,[])
    encoder_train.compile(optimizer,[loss_fn]*6)
    
    # add custom loss to metrics
    encoder_train.add_metric(kl_loss, name= "kl_loss") 
    encoder_train.add_metric(kl_loss1, name= "kl_loss1") 
    encoder_train.add_metric(kl_loss2, name= "kl_loss2") 
    
    
    encoder_train.summary()
    
    return encoder_train
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
                    annealing_duration= 2 # in epoches
                    annealing_begin = 2
                    annealing_ratio= max((epoch + steps_done/steps_per_epoch - annealing_begin),0) / annealing_duration
                    kl_loss_lamda = min(annealing_ratio,1)*np.ones((batch_size,1)) # linearly annealing kl loss lamda 
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
        mu1, logvar1 = encoder1.predict(data[0][0:16])
        mu2, logvar2 = encoder2.predict(data[1][0:16])
        
        z_parameters = _product_of_experts([mu1,mu2,logvar1,logvar2]) 
        z1_parameters= _product_of_experts([mu1,logvar1])
        z2_parameters= _product_of_experts([mu2,logvar2])
       
        z=_sampling(z_parameters)
        z1 = _sampling(z1_parameters)
        z2 = _sampling(z2_parameters)
        
        # reconstruct im
        for j,zi in enumerate([z,z1,z2]):
            for i,decoder in enumerate([decoder1,decoder2]):
                recon_imgs = decoder.predict(zi,steps=1)
                filename = 'recon_imgs'+str(i)+'/modality'+str(i)+'recon_z'+str(j)+'_%d_%d.png' % (self._epoch, self._steps)
                self.save_plot(recon_imgs, filename)
            
        # create image from linear interpolation over latent space
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
        for model in self._models:
                model.save_weights('model_checkpoints/'+model.name + suffix)


#!/usr/bin/env python3

import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import TensorBoard

from keras.losses import mean_absolute_error

#model_path = ""
model_path = "/kaggle/working/model_checkpoints/"


def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def main():
    
    encoders, decoders = create_models(n_channels=1)
    encoder_train = build_graph(encoders, decoders)

    try:
        initial_epoch = int(sys.argv[1])
    except (IndexError, ValueError):
        initial_epoch = config_initial_epoch

    epoch_format = '.{epoch:03d}.h5'

    if initial_epoch != 0:
        suffix = epoch_format.format(epoch=initial_epoch)
        encoders[0].load_weights(model_path+'encoder1' + suffix)
        encoders[1].load_weights(model_path+'encoder1' + suffix)
        decoders[0].load_weights(model_path+'decoder1' + suffix)
        decoders[1].load_weights(model_path+'decoder2' + suffix)


    checkpoint = ModelsCheckpoint(epoch_format, encoders[0],encoders[1], decoders[0],decoders[1])
    decoder_sampler = DecoderSnapshot()

    callbacks = [checkpoint, decoder_sampler]

    steps_per_epoch = NUM_SAMPLES // batch_size

    seed = np.random.randint(0,2**32 - 1,None,"int64")
    num_child = 1
    #img_loader = celeba_loader(batch_size, num_child=num_child, seed=seed)
    img_loader = mnist_loader(batch_size, num_child=num_child, seed=seed)
    enc_loader = encoder_loader(img_loader)
 
    models = [encoder_train]
    generators = [enc_loader]

    
#    metrics = [{'di_l': 1, 'di_l_t': 2, 'di_l_p': 3, 'di_a': 4, 'di_a_t': 7, 'di_a_p': 10}, {'de_l_t': 1, 'de_l_p': 2, 'de_a_t': 3, 'de_a_p': 5}, {'en_l': 0}]
    
    histories = fit_models(encoder_train, models, generators, batch_size,
                           steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                           epochs=epochs, initial_epoch=initial_epoch)
    with open('histories.pickle', 'wb') as f:
        pickle.dump(histories, f)



if __name__ == '__main__':
    main()
