from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import matplotlib.pyplot as plt # plotting

from skimage.io import imread # read in images

from skimage.segmentation import mark_boundaries # mark labels

from sklearn.metrics import roc_curve, auc # roc curve tools

from skimage.color import label2rgb

import numpy as np # linear algebra / matrices

# make the notebook interactive

from ipywidgets import interact, interactive, fixed 

import ipywidgets as widgets #add new widgets

from IPython.display import display

import inspect, json

import h5py

import skimage.transform

import scipy

import matplotlib as mpl

import os

import warnings

os.environ['KERAS_BACKEND'] = 'theano'

%matplotlib inline
from keras import backend as K

K.set_image_dim_ordering('th')

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D

from keras.layers import merge, Input

from keras.layers.core import ActivityRegularization

from keras.regularizers import l2

from keras.optimizers import SGD, Adam

from keras.utils import np_utils

MAKE_RGB_LAYER = False
def build_nd_fake_unet(in_shape, layers, depth,

                    conv_op, pool_op, upscale_op,

                   layer_size_fcn = lambda i: 3,

                   pool_size = 2,

                   dropout_rate = 0.0):

    inputs = Input(in_shape)

    

    first_layer = conv_op(depth,3)(conv_op(depth,3)(inputs))

    

    last_layer = first_layer

    

    conv_layers = []

    pool_layers = []

    

    for ilay in range(layers):

        # double filters

        pool_layers += [pool_op(pool_size)(last_layer)]

        lay_depth = depth*np.power(2,ilay+1)

        lay_kern_wid = layer_size_fcn(ilay)

        post_conv_step = conv_op(lay_depth, lay_kern_wid)(conv_op(lay_depth, lay_kern_wid)(pool_layers[-1]))

        if dropout_rate > 0: post_conv_step = Dropout(dropout_rate)(post_conv_step)

        

        conv_layers += [post_conv_step]

        

        last_layer = conv_layers[-1]



    # remove the last layer

    rev_layers = list(reversed(list(zip(range(layers),conv_layers[:-1]))))

    rev_layers += [(-1,first_layer)]

    

    for ilay, l_pool in rev_layers:

        lay_depth = depth*np.power(2,ilay+1)

        cur_up = upscale_op(pool_size)(last_layer)

        cur_merge = merge([cur_up, l_pool], mode='concat', concat_axis=1)

        cur_conv = conv_op(lay_depth, 3)(conv_op(lay_depth, 3)(cur_merge))

        last_layer = cur_conv

    

    

    out_conv = conv_op(1, 1, activation='tanh')(last_layer)

    ar_out = ActivityRegularization(l1=1e-4, l2=1e-2)(out_conv)

    model = Model(input=inputs, output=ar_out)

    

    return model



def build_2d_umodel(in_img, layers, depth = 8, lsf = lambda i: 3, pool_size = 3, dropout_rate = 0):

    conv_op = lambda n_filters, f_width, activation='relu', **kwargs: Convolution2D(n_filters, (f_width, f_width), activation=activation, border_mode='same', **kwargs)

    pool_op = lambda p_size: MaxPooling2D(pool_size=(p_size, p_size))

    upscale_op = lambda p_size: UpSampling2D(size=(p_size, p_size))

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        return build_nd_fake_unet(in_img.shape[1:], layers, depth, conv_op, pool_op, upscale_op,

                           layer_size_fcn = lsf, pool_size = pool_size, dropout_rate = dropout_rate)
with np.load('../input/depth_training_data.npz') as train_data_file:

    rgb_images = train_data_file['rgb_images']

    depth_maps = train_data_file['depth_maps']

    train_rgb_images = ((rgb_images.astype(np.float32)-127)/127)

    print(train_rgb_images.shape,depth_maps.shape)
rgbd_model = build_2d_umodel(rgb_images, layers = 5, depth = 4, pool_size = 2, 

                             dropout_rate = 0.25)

# overview of network

list(enumerate(map(lambda x: (x.name,('in:',x.input_shape[1:],'out:',x.output_shape[1:])),rgbd_model.layers)))
from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG

# Define model

try:

    vmod = model_to_dot(rgbd_model)

    vmod.write_svg('depth_net.svg')

    SVG('depth_net.svg')

except ImportError as ie:

    print('Cannot render Graph model',ie)
rgbd_model.summary()
def show_rgbd_maps(rgb_images, seg_images, count = 3, shuffle = True):

    batch_idx = np.array(range(rgb_images.shape[0]))

    if shuffle: np.random.shuffle(batch_idx)

    fig, ax_all = plt.subplots(2, count, figsize = (12,8))

    for pid, (c_raw_ax, c_flt_ax) in zip(batch_idx,ax_all.T):

        c_raw_ax.imshow(rgb_images[pid,:,:,:].swapaxes(0,2).swapaxes(0,1))

        c_raw_ax.set_title("Color Image:{}".format(pid))

        c_raw_ax.axis('off')

        c_flt_ax.imshow(seg_images[pid,0,:,:], cmap='gray', vmin = -1, vmax = 1)

        c_flt_ax.set_title("Depth Image:{}".format(pid))

        c_flt_ax.axis('off')
show_rgbd_maps(rgb_images, depth_maps,6)
from mpl_toolkits.mplot3d.axes3d import *

import matplotlib.pyplot as plt

from matplotlib import cm



def draw_surface_fig(color_arr, depth_arr):

    fig = plt.figure(figsize = (15,15))

    ax = Axes3D(fig)

    raw_shape = color_arr.shape

    xx,yy = np.meshgrid(range(raw_shape[1]),range(raw_shape[0]))

    # flip xx

    xx = xx[:,::-1]

    yy = yy[::-1,:]

    Zdata = depth_arr[::1,::1].astype(np.float32)

    Zdata -= Zdata.mean()

    Zdata /= -1*Zdata.std()

    ax.plot_surface(xx[::1,::1],yy[::1,::1],Zdata, rstride=1, cstride=1, 

                    facecolors=color_arr[::1,::1].astype(np.float32)/255,

                    linewidth=0, antialiased=True)

    ax.set_zlim3d((-1.5,1.5))

    #ax.axis('off')

    ax.view_init(70,45)

    return fig
rgbd_model.compile(optimizer=Adam(lr=1e-5), loss='mse')



if not os.path.exists('rgbd_deep_model.h5'):

    loss_history = []

else:

    rgbd_model.load_weights('rgbd_deep_model.h5') #dwi_weights_cnn96_02_06.h5')
from keras.preprocessing.image import ImageDataGenerator

batch_size = 20

datagen = ImageDataGenerator(

    rotation_range=5,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=True)



# compute quantities required for featurewise normalization

# (std, mean, and principal components if ZCA whitening is applied)

datagen.fit(train_rgb_images)
%%time

# a fast fit

# fit the model with the normalized images and the labels

big_depth = np.argsort(-np.std(depth_maps,(1,2,3)))[0:80]

for i in range(0):

    fit_history = rgbd_model.fit(train_rgb_images[big_depth,:,:,:],depth_maps[big_depth,:],

                                  batch_size = batch_size, nb_epoch = 20, shuffle = True, validation_split = 0.2)

    loss_history += [fit_history]
%%time

import os

fig_out_dir = 'depth_maps'

try:

    os.mkdir(fig_out_dir)

except:

    print(fig_out_dir,'already created')



    batch_size = 5

# fit the model with the normalized images and the labels

for i in range(0,2):

    

    plt.close()

    for j in [big_depth[1],1394,1973,1863]:

        pred_depth = rgbd_model.predict_on_batch(np.expand_dims(train_rgb_images[j],0))[0]

        fig = draw_surface_fig(rgb_images[j].swapaxes(0,2).swapaxes(0,1),

                              pred_depth[0])

        fig.savefig(os.path.join(fig_out_dir,'outdir_%02d_%04d.jpg' % (j,i)))

        fig.clf()

        plt.close()

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20,10))

        ax1.imshow(rgb_images[j].swapaxes(0,2).swapaxes(0,1))

        ax1.axis('off')

        ax2.imshow(depth_maps[j,0], vmin = -1, vmax = 1, cmap = 'bone')

        ax2.axis('off')

        ax2.set_title("Actual")

        ax3.imshow(pred_depth[0], vmin = -1, vmax = 1, cmap='bone')

        ax3.axis('off')

        ax3.set_title("Predicted")

        fig.savefig(os.path.join(fig_out_dir,'rgb_depth_%02d_%04d.jpg' % (j,i)))

        fig.clf()

        plt.close()

    

    fit_history = rgbd_model.fit_generator(

        datagen.flow(train_rgb_images, depth_maps, batch_size=batch_size),

        samples_per_epoch = 50, nb_epoch = 2)

    loss_history += [fit_history]

    
# predict the output layer of the images

prev_imgs = np.arange(train_rgb_images.shape[0])

np.random.shuffle(prev_imgs)

prev_imgs = prev_imgs[0:6]

pred_fimg = rgbd_model.predict_on_batch(train_rgb_images[prev_imgs])

pred_img = np.expand_dims(np.vstack(pred_fimg),1)

print(pred_img.shape)

show_rgbd_maps(rgb_images[prev_imgs], pred_img,6, shuffle = False)

show_rgbd_maps(rgb_images[prev_imgs], depth_maps[prev_imgs],6, shuffle = False)
epich = np.cumsum(np.concatenate([[1] * len(mh.epoch) for mh in loss_history]))

_ = plt.plot(epich,np.concatenate([mh.history['loss'] for mh in loss_history]),'b-')
rgbd_model.save_weights('rgbd_deep_model_%03d.h5' % i)
big_image = imread('../input/3d_scenes/img00003.tiff')[0:1024,0:1024,:3]

big_tensor = np.expand_dims(((big_image.astype(np.float32)-127)/127),0).swapaxes(3,1).swapaxes(2,3)

print(big_tensor.shape)
big_model = build_2d_umodel(big_tensor, layers = 5, depth = 4, pool_size = 2, 

                             dropout_rate = 0.25)

# overview of network

big_model.load_weights('rgbd_deep_model_%03d.h5' % i)

list(enumerate(map(lambda x: (x.name,('in:',x.input_shape[1:],'out:',x.output_shape[1:])),big_model.layers)))
%%time

pred_depth = big_model.predict_on_batch(big_tensor)[0]
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,10))

ax1.imshow(big_image)

ax1.axis('off')

ax2.imshow(pred_depth[0], vmin = -2, vmax = 2, cmap = 'bone')

ax2.axis('off')

ax2.set_title("Predicted")
fig = draw_surface_fig(big_image[::4,::4],pred_depth[0][::4,::4])