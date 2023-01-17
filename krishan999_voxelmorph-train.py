"""
train atlas-based alignment with CVPR2018 version of VoxelMorph 
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model

# project imports
vm_dir = "../input/cpvr-voxelmorph/cpvr_version/CPVR_version"
sys.path.append(os.path.join(vm_dir, 'src'))
sys.path.append(os.path.join(vm_dir, 'ext', 'medipy-lib'))
sys.path.append(os.path.join(vm_dir, 'ext', 'neuron'))
# print(sys.path)


import datagenerators
import networks
import losses


## some data prep
# Volume size used in our experiments. Please change to suit your data.
vol_size = (160, 192, 224)  

# prepare the data
# for the CVPR paper, we have data arranged in train/validate/test folders
# inside each folder is a /vols/ and a /asegs/ folder with the volumes
# and segmentations
base_data_dir = '../input/oasis-mri-npz-disc12/oasis_cross-sectional_disc12/'
# 获得数据集里所有的npz文件，生成一个list包含所有的数据名
train_vol_names = glob.glob(base_data_dir + '*.npz')
random.shuffle(train_vol_names)  # shuffle volume list

# load atlas from provided files. This atlas is 160x192x224.
atlas = np.load('../input/oasis-mri-npz-disc12/oasis_cross-sectional_disc12/OAS1_0001_MR1_mpr-1_anon.npz')
# 169x192x224 -> 1x160x192x224x1
atlas_vol = atlas['vol_data'][np.newaxis,...,np.newaxis]

print(len(train_vol_names))
def train(model='vm2', model_dir='../input/cpvr-voxelmorph/cpvr_version/CPVR_version/models', gpu_id=0, lr=1e-4, n_iterations=15000, reg_param=1.0, model_save_iter=1000, batch_size=1):
    """
    model training function
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param model_save_iter: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    """
    
    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # GPU handling
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # UNET filters for voxelmorph-1 and voxelmorph-2,
    # these are architectures presented in CVPR 2018
    nf_enc = [16, 32, 32, 32]
    if model == 'vm1':
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # prepare the model
    # in the CVPR layout, the model takes in [image_1, image_2] and outputs [warped_image_1, flow]
    # in the experiments, we use image_2 as atlas
    model = networks.unet(vol_size, nf_enc, nf_dec)
    # gradientLoss('L2') gets the L2 gradients
    model.compile(optimizer=Adam(lr=lr), 
                  loss=[losses.cc3D(), losses.gradientLoss('l2')],
                  loss_weights=[1.0, reg_param])

    # if you'd like to initialize the data, you can do it here:
    # model.load_weights(os.path.join(model_dir, '120000.h5'))

    # prepare data for training
    # example_gen返回一个generator
    train_example_gen = datagenerators.example_gen(train_vol_names)
    # 设置一个zero_flow，但是实质上在训练过程中不起作用
    zero_flow = np.zeros([batch_size, *vol_size, 3])

    # train. Note: we use train_on_batch and design out own print function as this has enabled 
    # faster development and debugging, but one could also use fit_generator and Keras callbacks.
    for step in range(0, n_iterations):

        # get data, generator returns a training example
        X = next(train_example_gen)[0]

        # train: train_on_batch run the single update and returns scalar loss or list of scalar losses
        train_loss = model.train_on_batch([X, atlas_vol], [atlas_vol, zero_flow])
        if not isinstance(train_loss, list):
            train_loss = [train_loss]

        # print the loss. 
        print_loss(step/2, 1, train_loss)

        # save model
#         if step % model_save_iter == 0:
#             model.save(os.path.join(model_dir, str(step) + '.h5'))


def print_loss(step, training, train_loss):
    """
    Prints training progress to std. out
    :param step: iteration number
    :param training: a 0/1 indicating training/testing
    :param train_loss: model loss at current iteration
    """
    s = str(step) + "," + str(training)

    if isinstance(train_loss, list) or isinstance(train_loss, np.ndarray):
        for i in range(len(train_loss)):
            s += "," + str(train_loss[i])
    else:
        s += "," + str(train_loss)

    print(s)
    # 一旦有结果立马输出
    sys.stdout.flush()
# train()
