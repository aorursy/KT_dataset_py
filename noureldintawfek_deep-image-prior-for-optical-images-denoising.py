# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:00:37 2020

@author: Noureldin Tawfek
"""

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Conv2DTranspose, Flatten, Reshape, \
    Lambda, LeakyReLU, Activation,concatenate,MaxPooling2D,UpSampling2D,Layer,InputSpec
from keras.regularizers import l2
from math import ceil

upsampling_method = "bilinear"
padding = "same" # "valid" or "same"
leaky_relu_alpha = 0.2
bn_mom=0.9
bn_eps=1e-6

resize_method = "lanczos3"
 



def paddings(input_s,strides,filter_s):
    
    input_w, input_h = input_s
    strides_w, strides_h = strides
    filter_w, filter_h = filter_s
    
    
    output_h = int(ceil(float(input_h) / float(strides[0])))
    output_w = int(ceil(float(input_w) / float(strides[1])))


    if input_h % strides[0] == 0:
        pad_along_height = max((filter_h - strides[0]), 0)
    else:
        pad_along_height = max(filter_h - (input_h % strides[0]), 0)
    if input_w % strides[1] == 0:
        pad_along_width = max((filter_w - strides[1]), 0)
    else:
        pad_along_width = max(filter_w - (input_w % strides[1]), 0)

    pad_top = pad_along_height // 2 #amount of  padding on the top
    pad_bottom = pad_along_height - pad_top # amount of  padding on the bottom
    pad_left = pad_along_width // 2             # amount of  padding on the left
    pad_right = pad_along_width - pad_left      # amount of  padding on the right
    
    return  (pad_top,pad_bottom,pad_left,pad_right)

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_output_shape_for(self, s):
        # "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        pad_top,pad_bottom,pad_left,pad_right = self.padding
        return tf.pad(x, [[0,0], [pad_top,pad_bottom], [pad_left,pad_right], [0,0] ], 'REFLECT')
    

def downsample_block(x, filters, kernelsize, name=''):

    pad_conv1 = paddings( (x.shape[2],x.shape[1]), (2,2), (kernelsize, kernelsize) )
    
    #x = ReflectionPadding2D(padding = pad_conv1)(x)
    x = Conv2D(filters, kernelsize , strides=2, padding=padding, name=name + 'conv1')(x)
    
    x = BatchNormalization(name=name + 'bn1')(x)
    x = LeakyReLU(leaky_relu_alpha,name=name + 'relu1')(x)
    
    pad_conv2 = paddings( (x.shape[2],x.shape[1]), (1,1), (kernelsize, kernelsize) )
    #x = ReflectionPadding2D(padding=pad_conv2)(x)
    x = Conv2D(filters, kernelsize , strides=1, padding=padding, name=name + 'conv2')(x)
    
    x = BatchNormalization( name=name + 'bn2')(x)
    x = LeakyReLU(leaky_relu_alpha,name=name + 'relu2')(x)
    
    
    return x

def upsample_block(x, filters, kernelsize, name=''):

    x = BatchNormalization(name=name + 'bn1')(x)
    
    pad_conv1 = paddings( (x.shape[2],x.shape[1]), (1,1), (kernelsize, kernelsize) )
    #x = ReflectionPadding2D(padding = pad_conv1)(x)
    x = Conv2D(filters, kernelsize , strides=1, padding=padding, name=name + 'conv1')(x)
    x = LeakyReLU(leaky_relu_alpha,name=name + 'relu1')(x)
    x = BatchNormalization(name=name + 'bn2')(x)
    
    pad_conv2 = paddings( (x.shape[2],x.shape[1]), (1,1), (kernelsize, kernelsize) )
    #x = ReflectionPadding2D(padding = pad_conv2)(x)
    x = Conv2D(filters, kernelsize , strides=1, padding=padding, name=name + 'conv2')(x)
    x = LeakyReLU(leaky_relu_alpha, name=name + 'relu2')(x)
    x = BatchNormalization(name=name + 'bn3')(x)
        
    x = UpSampling2D(interpolation = upsampling_method)(x)
    
    return x

def skip_block(x,filters, kernelsize,name = ""):

    pad_conv1 = paddings( (x.shape[2],x.shape[1]), (1,1), (kernelsize, kernelsize) )
 #  x = ReflectionPadding2D(padding = pad_conv1)(x)
    x = Conv2D(filters, kernelsize , strides=1, padding=padding, name=name + 'conv1')(x)
    
    x = BatchNormalization( name=name + 'bn1')(x)
    x = LeakyReLU(leaky_relu_alpha, name=name + 'relu1')(x)
    
    return x
    

def create_unet(down_filters, up_filters, skip_filters, kernel_d, kernel_u, kernel_s,input_shape,image_shape):
    x = Input(shape=input_shape, name='input_noise')
    mask = Input(shape=image_shape, name='mask')
    # create downsampling blocks 
    y = x
    downsampled_features = []
    for i, layer_filters in enumerate(down_filters):
        y = downsample_block(y,down_filters[i],kernel_d[i], name = "downsample_block"+str(i))
        downsampled_features.append(y)
        print(y.shape)
        
    for i, layer_filters in enumerate(up_filters):
        if skip_filters[-(i+1)] != 0: # skip connection 
            skipped = skip_block(downsampled_features[-(i+1)],skip_filters[-(i+1)],kernel_s[-(i+1)], "skip_block"+str(i))
            y = concatenate([y, skipped], axis=3) # concetenate feature over the channels axis 
        y = upsample_block(y,up_filters[i],kernel_u[i] , name = "upsample_block"+str(i))
        print(y.shape)
    
    clean_image = Conv2D(1, 1)(y)
    
    
    
    unet = Model([x,mask], [clean_image,clean_image,clean_image], name="unet")

    return unet

def create_superresolution_unet(down_filters, up_filters, skip_filters, kernel_d, kernel_u, kernel_s,input_shape,resampling_factor):
    
    height =input_shape[1]
    width =input_shape[0]
    
    unet = create_unet(down_filters, up_filters, skip_filters, kernel_d, kernel_u, kernel_s,input_shape)
    
    x = Input(shape=input_shape, name='enc_input')
    hr_images  = unet(x)
    lr_images = tf.image.resize(hr_images, size= [height,width], method=resize_method, preserve_aspect_ratio=False,antialias=False, name="resizing")
    SR_unet = Model(x,lr_images,name = "SR_unet")
    
    return SR_unet, unet 
    




nbins = 100
num_of_looks = 8
gamma_sample_size = 100000000

def compute_probs(data, n=10): 
    h, e = np.histogram(data, n)
    p = h/data.shape[0]
    return e, p

gamma_sample = np.random.gamma(num_of_looks, scale=1/num_of_looks, size= (100000000,) ) # gamma distribution approximation
gamma_sample_max = np.max(gamma_sample)
gamma_sample_min = np.min(gamma_sample)



# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:38:57 2020

@author: Noureldin Tawfek
"""

import pickle

import tensorflow_probability as tfp 

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from PIL import Image
import matplotlib.pyplot as plt
import math

import numpy as np
iterations  = 20000
save_img_every_n_iteration= 100
save_model_every_n_iteration = 300
#image_path = "/denoising/F16_GT.png"
image_path ='/kaggle/input/denoising3/runway06.png'
input_noise_channels= 32
lr = 0.01 # learning rate 
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# optimizer = tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate=lr, preconditioner_decay_rate=0.99,burnin=25)
load_weights = False 
normal_noise = True #Perturbing the input x with an additive Gaussian noise with mean zero and standard deviation 
normal_noise_std = 1/30.0
grayscale = True

from tensorflow.python.ops import math_ops

noise_type = "speckle" # noise_type can be either "gaussian" or "speckle" (drawed from a gamma distribution)
log_data = True

from math import log10, sqrt 

def _load_image(f):
    im = Image.open(f)
    print(np.asarray(im).shape)
    return np.asarray(im)

def weighted_mse_loss(weights):
    def loss(y_true, y_pred):
       return tf.reduce_sum(weights*tf.pow(y_true-y_pred, 2))/(512*512)
    return loss

def mse_loss(base_content, target):
    return 



def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def PSNR_mse(mse):
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def vr_loss(y_true, y_pred):
    return 0.0009*0*tf.image.total_variation(y_pred)

def kl_div(gamma_sample,speckle):
    
    speckle = tf.keras.backend.flatten(speckle)
    print(speckle.shape)
    gamma_sample = tf.reshape(gamma_sample , (gamma_sample_size,))

    gamma_probs= tf.histogram_fixed_width(gamma_sample ,[gamma_sample_min,gamma_sample_max], nbins=nbins, name=None)/gamma_sample_size
    speckle_probs= tf.histogram_fixed_width(speckle ,[gamma_sample_min,gamma_sample_max], nbins=nbins, name=None)/tf.shape(speckle)[0]
    
    kl_loss = tf.keras.losses.KLDivergence()
    kl_div = kl_loss(speckle_probs, gamma_probs) 
    return 0*kl_div # kl_divergence

# local homogenity index

def calculate_LHI(img,num_of_looks):
    img = img[:,:,0]
    speckle_var = math.sqrt(1/num_of_looks)
    print(img.shape)
    neigh_size = 10
    pad_width = neigh_size//2
    rpatch = pad_width
    rows_0, cols_0 = img.shape
    img = np.pad(img, pad_width, mode='reflect')
    rows, cols = img.shape
    LHI = np.zeros(img.shape)
    LHI_img = Image.fromarray(LHI)
    LHI_img.show()
    a = []
    for i in range(rpatch,rows-rpatch):
        for j in range(rpatch,cols-rpatch):
            curr_patch = img[i-rpatch:i+rpatch+1,j-rpatch:j+rpatch+1].flatten()
            a.append(curr_patch)
    a = np.stack(a)
    mean = np.mean(a,axis = 1).reshape((rows_0,cols_0))
    var  = np.var(a,axis = 1 ).reshape((rows_0,cols_0))
    LHI = (var-(mean*speckle_var)**2)/((1+speckle_var**2)*var)
    LHI = np.clip(LHI,0,None)
    Image.fromarray((LHI*255).astype('uint8')).save("LHI.png")
    return LHI

def weighted_total_variation(LHI, images):

  
    ndims = images.get_shape().ndims

    if ndims == 3:
      pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
      pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
      sum_axis = None
    elif ndims == 4:
      pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
      pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
      sum_axis = [1, 2, 3]
    else:
      raise ValueError('\'images\' must be either 3 or 4-dimensional.')
    weights = 1-LHI
    pixel_dif1= math_ops.abs(pixel_dif1)*weights
    pixel_dif2= math_ops.abs(pixel_dif2)*weights
    tot_var = (
        math_ops.reduce_sum(math_ops.abs(pixel_dif1), axis=sum_axis) +
        math_ops.reduce_sum(math_ops.abs(pixel_dif2), axis=sum_axis))
    return tot_var



def save_image(image,filename):
    #image = image * 255.
    image = image.astype('uint8')
    
    if grayscale==True :
        Image.fromarray(image[:,:,0]).save(filename)
    else :
        Image.fromarray(image).save(filename)
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype('uint8')

def log_data_fidilty(y_true, y_pred):
    return tf.reduce_sum(y_pred + tf.exp(y_true-y_pred))

def main():

    print("Eager execution: {}".format(tf.executing_eagerly()))

    # loading image 
    clean_image = _load_image(image_path)
    if grayscale==True :
        clean_image = rgb2gray(clean_image)[:,:,None].astype('uint8')
    print(clean_image.shape)
    image_shape =  clean_image.shape
    sigma = 25
    if noise_type == "gaussian" :
        noised_image = clean_image + np.random.normal(scale=sigma, size=image_shape)
    elif noise_type == "speckle":
        speckle_noise = np.random.gamma(num_of_looks, scale=1/num_of_looks, size=image_shape)
        noised_image = clean_image*speckle_noise
    else :
        noised_image = clean_image
    noised_image = np.clip(noised_image,0,255)
    clean_image = np.clip(clean_image,0,255) # normalize the image
    LHI = calculate_LHI(noised_image,num_of_looks)
    
    psnr = PSNR(clean_image,noised_image)
    print("psnr between the orginal and denoised image : %d " % (psnr))
    # 
    mask = np.random.binomial(1, 0.3, size=clean_image.shape)
    save_image(clean_image,"original.png")
    # creating noise input 
    save_image(noised_image,"speckled.png")
    save_image(np.clip(speckle_noise,0,255),"test3.png")
    
    

    input_noise = np.random.uniform(low=0.0, high=0.1, size=(image_shape[0],image_shape[1],input_noise_channels))
    
    # defining model parameters and building the model
    down_filters =  [128, 128, 128, 128, 128]
    up_filters = [128, 128, 128, 128, 128]
    skip_filters = [4, 4, 4, 4, 4]
    kernel_d = [3, 3, 3 , 3, 3]
    kernel_u = [3, 3, 3 , 3, 3]
    kernel_s = [1, 1, 1, 1, 1]
    model = create_unet(down_filters, up_filters, skip_filters, kernel_d, kernel_u, kernel_s,input_noise.shape,image_shape)
    if load_weights:
        model.load_weights("model.h5")
    input_noise = input_noise[np.newaxis,...]
    down_filters, up_filters, skip_filters, kernel_d, kernel_u, kernel_s
    model.compile(loss=["mse",kl_div,vr_loss], optimizer=optimizer)
    model.summary()
    
    

    

    history_iterations = []
    history_loss = []
    history_psnr = []

    for iteration in range(iterations):
        # Perturbing the input x with an additive Gaussian noise with mean zero and standard deviation σp
        model_input = input_noise 
        noised_patch = noised_image[None, :, :, :]
        if log_data == True: 
            noised_patch =  np.log1p(noised_image[None, :, :, :])*255
        
        gamma_batch= gamma_sample.reshape((1,gamma_sample_size))
        if normal_noise:
            model_input += np.random.normal(0, normal_noise_std , (image_shape[0], image_shape[1], input_noise_channels))
        # calculating loss     
        loss = model.train_on_batch( [model_input,mask[None, :, :, :]] , [ noised_patch,gamma_batch,LHI[np.newaxis,:]])
        
        print("iteration :" + str(iteration) +",loss :" + str(loss) )
        # save image and history 
        if iteration % save_img_every_n_iteration==0 :
            output_image,speckle = model.predict([model_input,noised_patch])[0:2]
            if log_data == True: 
                output_image = np.exp(output_image/255)-1
                speckle = np.exp(speckle/255)-1
            output_image = np.clip(output_image,0,255)
            print(speckle.shape)
            speckle = np.clip(speckle,0,255)
            
            save_image(speckle[0] , "speckle"+str(iteration)+".png")
            #
            history_loss.append(loss)
            psnr = PSNR(clean_image,output_image)
            history_psnr.append(psnr)
            history_iterations.append(iteration)
            filename = ('itr%d_PNSR%.2f.png' % (iteration, psnr))
            save_image(output_image[0] , filename)

            #save_image(output_image[0].numpy() , filename)
        if (iteration+1) % save_model_every_n_iteration==0:
             model.save_weights("model.h5")
        #grads = tape.gradient(loss, model.trainable_variables)
        #optimizer.apply_gradients(zip(grads, model.trainable_variables))    
        #psnr = PSNR(clean_image,output_image)
        
    history = {}
    history["itr"] = history_iterations
    history["mse"] = history_loss
    history["psnr"] = history_psnr
    with open('histories.pickle', 'wb') as f:
        pickle.dump(history, f)
    
        
if __name__ == '__main__':
    main()

        
     
