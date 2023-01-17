%%capture
!git clone https://gist.github.com/belkhir-nacim/5230ccfcab05f30c35abb03444f6a216 dataset_util
!pip install pytorch-lightning==0.7.6
!pip install git+https://github.com/belkhir-nacim/generative_model_toolbox
!pip install fastai2
import functools
from enum import IntEnum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import argparse
from dataset_util.kaggle_textile_texuture_dataset import TextureDataset
import os
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
from generative_models_toolbox.utils.device import Cudafy
cudafy = Cudafy(0)
from fastai2.vision.all import *

# Optional dependency to visualize image gradient
import kornia
from typing import List
import math
from torch.nn.init import _calculate_correct_fan


laplace_filter = partial(kornia.filters.laplacian, kernel_size=3)
gradient_filter = kornia.filters.sobel

def visualize_filter(image, filter_func, ax=None,title=None):
    filt = filter_func(image.unsqueeze(0))
    # Normalizing to [0, 1] range
    filt -= filt.min()
    filt /= filt.max()
    img = kornia.tensor_to_image(filt[0])
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    ax.imshow(img,cmap='gray')
    if title is not None: ax.set_title(title)
    return ax

#Helper function
def decode_prediction(learner, inp):
    y_hat = learn.model(cudafy(inp))
    return TensorImage(y_hat.transpose(0, 1).reshape(1, 256, 256))

import torch
from torch import nn
import torch.nn.functional as F
from typing import List
import math


def siren_init(tensor, use_this_fan_in=None):
    """
        Siren initalization of a tensor. To initialize a nn.Module use 'apply_siren_init'. 
        It's equivalent to torch.nn.init.kaiming_uniform_ with mode = 'fan_in'
        and the same gain as the 'ReLU' nonlinearity
    """
    if use_this_fan_in is not None:
        fan_in = use_this_fan_in
    else:
        fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
    bound = math.sqrt(6.0 / fan_in)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def apply_siren_init(layer: nn.Module):
    """
        Applies siren initialization to a layer
    """
    siren_init(layer.weight)
    if layer.bias is not None:
        fan_in = nn.init._calculate_correct_fan(layer.weight, "fan_in")
        siren_init(layer.bias, use_this_fan_in=fan_in)


class Siren(nn.Module):
    """
        Siren activation
        https://arxiv.org/abs/2006.09661
    """

    def __init__(self, w0=1):
        """
            w0 comes from the end of section 3
            it should be 30 for the first layer
            and 1 for the rest
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def extra_repr(self):
        return "w0={}".format(self.w0)


def siren_layer(in_features, out_features, bias=True, w0=1):
    """
        Siren Layer - it's a modified linear layer with sine activation
    """
    layer = nn.Sequential(nn.Linear(in_features, out_features, bias), Siren(w0))
    apply_siren_init(layer[0])
    return layer


def siren_model(dimensions: List[int]):
    """
        Siren model as presented in the paper. It's a sequence of linear layers followed by the Siren activation
    """
    first_layer = siren_layer(dimensions[0], dimensions[1], w0=30)
    other_layers = []
    for dim0, dim1 in zip(dimensions[1:-1], dimensions[2:]):
        other_layers.append(siren_layer(dim0, dim1))
    return nn.Sequential(first_layer, *other_layers)
def run_xp_trial():

    dl = DataLoader(TextureDataset('/kaggle/input/textiledefectdetection',train=True, patch_size= 64, keep_angles=False,  keep_defects=False, sub_sample=2),shuffle=True, batch_size=1)
    img_ground_truth,_,_ = next(iter(dl))
    img_ground_original = img_ground_truth.squeeze(0)
    img_ground_truth = img_ground_truth.squeeze(0).permute(1,2,0)
    
    pipe = Pipeline([transforms.ToPILImage(), transforms.Resize(256), transforms.ToTensor()])
    image = pipe(img_ground_original)
    print(image.shape)
    fig, axes = plt.subplots(1,3,figsize=(15,8))
    visualize_filter(image, lambda x:x,ax=axes[0],title='original' )
    visualize_filter(image, gradient_filter,ax=axes[1],title='gradient')
    visualize_filter(image, laplace_filter,ax=axes[2],title='laplacian')
    
    y = image.reshape(1, -1).transpose(0, 1)
    g0, g1 = torch.meshgrid([torch.arange(-1, 1, step=2/256), torch.arange(-1, 1, step=2/256)])
    x = torch.cat([g0.flatten().unsqueeze(1), g1.flatten().unsqueeze(1)], dim=1)
    x = x.float()
    for coord, pixel_value in zip(x, y):
        c = ( 128 * (1 + coord)).long()
        assert (image[:, c[0], c[1]] == pixel_value).all(), "Pixel values do not match"
    from torch.utils.data import TensorDataset, random_split
    dset = TensorDataset(cudafy(x), cudafy(y))
    val_pct = 0.2
    val_len = int(len(dset)*val_pct)
    lengths = [len(dset)-val_len, val_len]
    train_dset, val_dset = random_split(dset, lengths)
    dls = DataLoaders(DataLoader(train_dset, bs=256), DataLoader(val_dset, bs=4096))
    learn = Learner(dls,  cudafy(siren_model([2, 256, 128, 64, 32, 1])),  loss_func=MSELossFlat(),opt_func=ranger)
    print(learn.model)
    learn.fit_flat_cos(5, lr=1e-3)
    
    fig, axes = plt.subplots(1,2,figsize=(15,8))
    visualize_filter(image, lambda x:x,ax=axes[0],title='original' )
    visualize_filter(decode_prediction(cudafy(learn), cudafy(x)), lambda x:x,ax=axes[1], title='decoded')

    fig, axes = plt.subplots(1,3,figsize=(15,8))
    visualize_filter(image, laplace_filter,ax=axes[0],title='original laplacian' )
    visualize_filter(decode_prediction(cudafy(learn), cudafy(x)), laplace_filter,ax=axes[1], title='decoded laplacian')
    tmp = laplace_filter(image.unsqueeze(0))  - laplace_filter(decode_prediction(cudafy(learn), cudafy(x)).unsqueeze(0) )
    visualize_filter(tmp , lambda x :x,ax=axes[2], title='diff')


    fig, axes = plt.subplots(1,3,figsize=(15,8))
    visualize_filter(image, gradient_filter,ax=axes[0],title='original gradient' )
    visualize_filter(decode_prediction(cudafy(learn), cudafy(x)), gradient_filter,ax=axes[1], title='decoded gradient')
    tmp = gradient_filter(image.unsqueeze(0))  - gradient_filter(decode_prediction(cudafy(learn), cudafy(x)).unsqueeze(0) )
    visualize_filter(tmp , lambda x :x,ax=axes[2], title='diff')
dl = DataLoader(TextureDataset('/kaggle/input/textiledefectdetection',train=False, patch_size= 64, keep_angles=False,  keep_defects=True, sub_sample=10),shuffle=True, batch_size=1)

img_ground_truth = None
for data,angle,label in dl:
    if label[0].item()==3:
        img_ground_truth = data
    else:
        pass
# img_ground_truth,_,_ = next(iter(dl)) if train == True
img_ground_original = img_ground_truth.squeeze(0)
img_ground_truth = img_ground_truth.squeeze(0).permute(1,2,0)
pipe = Pipeline([transforms.ToPILImage(), transforms.Resize(256), transforms.ToTensor()])
image = pipe(img_ground_original)
print(image.shape)
fig, axes = plt.subplots(1,3,figsize=(15,8))
visualize_filter(image, lambda x:x,ax=axes[0],title='original' )
visualize_filter(image, gradient_filter,ax=axes[1],title='gradient')
visualize_filter(image, laplace_filter,ax=axes[2],title='laplacian')
y = image.reshape(1, -1).transpose(0, 1)
y.shape
g0, g1 = torch.meshgrid([torch.arange(-1, 1, step=2/256), torch.arange(-1, 1, step=2/256)])
x = torch.cat([g0.flatten().unsqueeze(1), g1.flatten().unsqueeze(1)], dim=1)
x = x.float()
x.shape
for coord, pixel_value in zip(x, y):
    c = ( 128 * (1 + coord)).long()
    assert (image[:, c[0], c[1]] == pixel_value).all(), "Pixel values do not match"
from torch.utils.data import TensorDataset, random_split
dset = TensorDataset(cudafy(x), cudafy(y))
val_pct = 0.005
val_len = int(len(dset)*val_pct)
lengths = [len(dset)-val_len, val_len]
train_dset, val_dset = random_split(dset, lengths)
dls = DataLoaders(DataLoader(train_dset, bs=256), DataLoader(val_dset, bs=4096))
learn = Learner(dls,  cudafy(siren_model([2, 256, 128, 64, 32, 1])),  loss_func=MSELossFlat(),opt_func=ranger)
learn.model
learn.fit_flat_cos(150, lr=1e-3)
fig, axes = plt.subplots(1,2,figsize=(15,8))
visualize_filter(image, lambda x:x,ax=axes[0],title='original' )
visualize_filter(decode_prediction(learn, cudafy(x)), lambda x:x,ax=axes[1], title='decoded')

fig, axes = plt.subplots(1,3,figsize=(15,8))
visualize_filter(image, laplace_filter,ax=axes[0],title='original laplacian' )
visualize_filter(decode_prediction(learn, cudafy(x)), laplace_filter,ax=axes[1], title='decoded laplacian')
tmp = cudafy(laplace_filter(image.unsqueeze(0)))  - laplace_filter(decode_prediction(learn, cudafy(x)).unsqueeze(0) )
visualize_filter(tmp , lambda x :x,ax=axes[2], title='diff')


fig, axes = plt.subplots(1,3,figsize=(15,8))
visualize_filter(image, gradient_filter,ax=axes[0],title='original gradient' )
visualize_filter(decode_prediction(learn, cudafy(x)), gradient_filter,ax=axes[1], title='decoded gradient')
tmp = cudafy(gradient_filter(image.unsqueeze(0)))  - gradient_filter(decode_prediction(learn, cudafy(x)).unsqueeze(0) )
visualize_filter(tmp , lambda x :x,ax=axes[2], title='diff')

# fig, axes = plt.subplots(1,2,figsize=(15,8))
# visualize_filter(image, lambda x:x,ax=axes[0],title='original' )
# visualize_filter(decode_prediction(cudafy(learn), cudafy(x)), lambda x:x,ax=axes[1], title='decoded')

# fig, axes = plt.subplots(1,3,figsize=(15,8))
# visualize_filter(image, laplace_filter,ax=axes[0],title='original laplacian' )
# visualize_filter(decode_prediction(cudafy(learn), cudafy(x)), laplace_filter,ax=axes[1], title='decoded laplacian')
# tmp = laplace_filter(image.unsqueeze(0))  - laplace_filter(decode_prediction(cudafy(learn), cudafy(x)).unsqueeze(0) )
# visualize_filter(tmp , lambda x :x,ax=axes[2], title='diff')


# fig, axes = plt.subplots(1,3,figsize=(15,8))
# visualize_filter(image, gradient_filter,ax=axes[0],title='original gradient' )
# visualize_filter(decode_prediction(cudafy(learn), cudafy(x)), gradient_filter,ax=axes[1], title='decoded gradient')
# tmp = gradient_filter(image.unsqueeze(0))  - gradient_filter(decode_prediction(cudafy(learn), cudafy(x)).unsqueeze(0) )
# visualize_filter(tmp , lambda x :x,ax=axes[2], title='diff')
# Number of pixels on the image
256*256*3

# Number of parameters on the model
print('Number of parameters of the model {}'.format(sum([p.numel() for p in learn.model.parameters()])))
from typing import List
def relu_model(dimensions: List[int]):
    """
        Sequence of linear layers followed by ReLU
    """
    layers = []
    for dim0, dim1 in zip(dimensions[:-1], dimensions[1:]):
        layers.append(nn.Linear(dim0, dim1))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)
learn = Learner(dls,  cudafy(relu_model([2, 256, 128, 64, 32, 1])),  loss_func=MSELossFlat(), opt_func=ranger )
learn.model
learn.fit_flat_cos(150, lr=1e-3)
fig, axes = plt.subplots(1,2,figsize=(15,8))
visualize_filter(image, lambda x:x,ax=axes[0],title='original' )
visualize_filter(decode_prediction(learn, cudafy(x)), lambda x:x,ax=axes[1], title='decoded')

fig, axes = plt.subplots(1,3,figsize=(15,8))
visualize_filter(image, laplace_filter,ax=axes[0],title='original laplacian' )
visualize_filter(decode_prediction(learn, cudafy(x)), laplace_filter,ax=axes[1], title='decoded laplacian')
tmp = cudafy(laplace_filter(image.unsqueeze(0)))  - laplace_filter(decode_prediction(learn, cudafy(x)).unsqueeze(0) )
visualize_filter(tmp , lambda x :x,ax=axes[2], title='diff')


fig, axes = plt.subplots(1,3,figsize=(15,8))
visualize_filter(image, gradient_filter,ax=axes[0],title='original gradient' )
visualize_filter(decode_prediction(learn, cudafy(x)), gradient_filter,ax=axes[1], title='decoded gradient')
tmp = cudafy(gradient_filter(image.unsqueeze(0)))  - gradient_filter(decode_prediction(learn, cudafy(x)).unsqueeze(0) )
visualize_filter(tmp , lambda x :x,ax=axes[2], title='diff')

# fig, axes = plt.subplots(1,2,figsize=(15,8))
# visualize_filter(image, lambda x:x,ax=axes[0],title='original' )
# visualize_filter(decode_prediction(cudafy(learn), cudafy(x)), lambda x:x,ax=axes[1], title='decoded')

# fig, axes = plt.subplots(1,3,figsize=(15,8))
# visualize_filter(image, laplace_filter,ax=axes[0],title='original laplacian' )
# visualize_filter(decode_prediction(cudafy(learn), cudafy(x)), laplace_filter,ax=axes[1], title='decoded laplacian')
# tmp = laplace_filter(image.unsqueeze(0))  - laplace_filter(decode_prediction(cudafy(learn), cudafy(x)).unsqueeze(0) )
# visualize_filter(tmp , lambda x :x,ax=axes[2], title='diff')


# fig, axes = plt.subplots(1,3,figsize=(15,8))
# visualize_filter(image, gradient_filter,ax=axes[0],title='original gradient' )
# visualize_filter(decode_prediction(cudafy(learn), cudafy(x)), gradient_filter,ax=axes[1], title='decoded gradient')
# tmp = gradient_filter(image.unsqueeze(0))  - gradient_filter(decode_prediction(cudafy(learn), cudafy(x)).unsqueeze(0) )
# visualize_filter(tmp , lambda x :x,ax=axes[2], title='diff')