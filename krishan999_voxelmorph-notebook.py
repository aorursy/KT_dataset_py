# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../"))
print(os.listdir("../input"))
# Print the current path
print(os.path.abspath(os.curdir))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
vm_dir = "../input/voxelmorph-v3/voxelmorph_3/voxelmorph_3/voxelmorph-master/"
sys.path.append(os.path.join(vm_dir, 'src'))
sys.path.append(os.path.join(vm_dir, 'ext', 'medipy-lib'))
sys.path.append(os.path.join(vm_dir, 'ext', 'neuron'))
sys.path.append(os.path.join(vm_dir, 'ext', 'pynd-lib'))
sys.path.append(os.path.join(vm_dir, 'ext', 'pytool-lib'))
print(os.listdir('../input'))
print(os.listdir(vm_dir))
import medipy
import datagenerators
from medipy.metrics import dice
import networks
print(sys.path)
def print_image(vol):
    depth, height, width = vol.shape
    fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 10))
    ax1.imshow(np.mean(vol[:, :, :], axis=0))
    ax2.imshow(np.mean(vol[:, :, :], axis=1))
    ax3.imshow(np.mean(vol[:, :, :], axis=2))
    ax4.imshow(vol[depth//2, :, :])
    ax5.imshow(vol[:, height//2, :])
    ax6.imshow(vol[:, :, width//2])
def compare_images(y_pred, X_vol, atlas_vol):
    # show the mean of the input_test_data in each dimension
    fig, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(20,10))
    ax1.imshow(np.mean(y_pred[0,:, :, :,0], axis=0))
    ax1.set_title('pred')
    ax2.imshow(np.mean(y_pred[0,:, :, :,0], axis=1))
    ax2.set_title('pred')
    ax3.imshow(np.mean(y_pred[0,:, :, :,0], axis=2))
    ax3.set_title('pred')
    ax4.imshow(np.mean(X_vol[0,:, :, :,0], axis=0))
    ax4.set_title('moving-image')
    ax5.imshow(np.mean(X_vol[0,:, :, :,0], axis=1))
    ax5.set_title('moving-image')
    ax6.imshow(np.mean(X_vol[0,:, :, :,0], axis=2))
    ax6.set_title('moving-image')
    ax7.imshow(np.mean(atlas_vol[0,:, :, :,0], axis=0))
    ax7.set_title('atlas-image')
    ax8.imshow(np.mean(atlas_vol[0,:, :, :,0], axis=1))
    ax8.set_title('atlas-image')
    ax9.imshow(np.mean(atlas_vol[0,:, :, :,0], axis=2))
    ax9.set_title('atlas-image')
def meshgridnd_like(in_img,
                    rng_func=range):
    new_shape = list(in_img.shape)
    all_range = [rng_func(i_len) for i_len in new_shape]
    # np.swapaxes()交换坐标轴：由(y, x, z)变为(x, y, z)
    return tuple([x_arr.swapaxes(0, 1) for x_arr in np.meshgrid(*all_range)])
from mpl_toolkits.mplot3d import axes3d
def print_flow(vol_size, flow):
    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
    DS_FACTOR = 16
    # c_xx, x_yy, c_zz are the x, y, z array for 10x12x14=1980 points
    c_xx, c_yy, c_zz = [x.flatten()
                        for x in 
                        meshgridnd_like(flow[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR, 0])]

    get_flow = lambda i: flow[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR, i].flatten()

    fig = plt.figure(figsize = (10, 10))
    ax = fig.gca(projection='3d')

    ax.quiver(c_xx,
              c_yy,
              c_zz,
              get_flow(0),
              get_flow(1), 
              get_flow(2), 
              length=0.9,
              normalize=True)
# Set up the Network
# We use the voxelmorph-2 here, with dec [32,32,32,32,32,16,16,3]
# vol_size = (160, 192, 224) #(height, width, depth)
# nf_enc = [16, 32, 32, 32]
# nf_dec = [32, 32, 32, 32, 32, 16, 16, 3]
vol_size=(160,192,224)
nf_enc=[16,32,32,32]
nf_dec=[32,32,32,32,32,16,16]
# load Atals data
labels = sio.loadmat(os.path.join(vm_dir, 'data', 'labels.mat'))['labels'][0]
atlas = np.load(os.path.join(vm_dir, 'data', 'atlas_affine_vol.npz'))
# Print what is inside the atlas
for key, value in atlas.items():
    print(key)
atlas_vol = atlas['vol_data']
# atlas_seg = atlas['seg']
# Expand the dimension of the input: the first number is the number of the data, and the last one is the channel number
atlas_vol = np.reshape(atlas_vol, (1,)+atlas_vol.shape+(1,)) # 1x160x192x224x1
print(atlas_vol.shape)
print_image(atlas_vol[0,:,:,:,0])
test_npz = np.load(vm_dir + 'data/test_affine_1.npz')
for k, v in test_npz.items():
    print(k)
seg = test_npz['seg']
print(seg.shape)
print_image(seg)
import glob
test_affine_names = glob.glob("../input/test-affine-data/test_affine/test_affine/*.npz")
test_affine_movings = []
test_affine_fixeds = []
test_affine_segs = []
for i in range(len(test_affine_names)):
    test_affine_movings.append(np.load(test_affine_names[i])['moving'][np.newaxis,...,np.newaxis])
    test_affine_fixeds.append(np.load(test_affine_names[i])['fixed'][np.newaxis,...,np.newaxis])
    test_affine_segs.append(np.load(test_affine_names[i])['seg'])
# test_affine_movings = np.concatenate(test_affine_movings, axis=0)
# test_affine_fixeds = np.concatenate(test_affine_fixeds, axis=0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))
gpu = '/gpu:0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# load weights of model
model_name = vm_dir + 'models/255.h5'
with tf.device(gpu):
    net = networks.cvpr2018_net(vol_size, nf_enc, nf_dec)
    # net.load_weights('../models/' + model_name + '/' + str(iter_num) + '.h5')
    net.load_weights(model_name)
X_vol = np.load(vm_dir + 'data/test_affine_vol.npz')['vol_data']
X_vol = X_vol[np.newaxis, ... , np.newaxis]
print(X_vol.shape)
with tf.device(gpu):
    pred = []
    for i in range(len(test_affine_movings)):
        pred.append(net.predict([test_affine_movings[i], test_affine_fixeds[i]]))
for i in range(len(test_affine_fixeds)):
    compare_images(y_pred=pred[i][0], X_vol=test_affine_movings[i], atlas_vol=test_affine_fixeds[i])
for i in range(len(test_affine_fixeds)):
    print_flow(vol_size=vol_size, flow=pred[i][1][0])
for i in range(len(test_affine_fixeds)):
    diff_ = np.abs(test_affine_fixeds[i]-pred[i][0])
    print_image(diff_[0,:,:,:,0])
