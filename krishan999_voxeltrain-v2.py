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
vm_dir = "../input/voxelmorphv2/voxelmorph_v2/voxelmorph_v2"
sys.path.append(os.path.join(vm_dir, 'src'))
sys.path.append(os.path.join(vm_dir, 'ext', 'medipy-lib'))
sys.path.append(os.path.join(vm_dir, 'ext', 'neuron'))
sys.path.append(os.path.join(vm_dir, 'ext', 'pynd-lib'))
sys.path.append(os.path.join(vm_dir, 'ext', 'pytool-lib'))
print(sys.path)

data_path = "../input/oasis-mri-npz-disc12/oasis_cross-sectional_disc12"
import datagenerators
import networks
import losses
from train import train
train(data_path, atlas_file='../input/voxeltrainv2/voxelmorph_v2/voxelmorph_v2/data/atlas_norm.npz', gpu_id='0')
