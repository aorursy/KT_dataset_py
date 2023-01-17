## Load Libraries 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import gc

import matplotlib.pyplot as plt

import torch.nn.functional as F

import os

# Any results you write to the current directory are saved as output.

import torch

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms,models

from tqdm import tqdm_notebook as tqdm



## This library is for augmentations .

from albumentations import (

    PadIfNeeded,

    HorizontalFlip,

    VerticalFlip,    

    CenterCrop,    

    Crop,

    Compose,

    Transpose,

    RandomRotate90,

    ElasticTransform,

    GridDistortion, 

    OpticalDistortion,

    RandomSizedCrop,

    OneOf,

    CLAHE,

    RandomBrightnessContrast,    

    

    RandomGamma,

    ShiftScaleRotate ,

    GaussNoise,

    Blur,

    MotionBlur,   

    GaussianBlur,

)



import warnings

warnings.filterwarnings('ignore')
##Path for data 

PATH = '../input/bengaliai-cv19/'
## Create Data from Parquet file mixing the methods of @hanjoonzhoe and @Iafoss



## Create Crop Function @Iafoss



HEIGHT = 137

WIDTH = 236

SIZE = 128



def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=SIZE, pad=16):

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    #cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    #remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    #make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(img,(size,size))



def Resize(df,size=128):

    resized = {} 

    df = df.set_index('image_id')

    for i in tqdm(range(df.shape[0])):

       # image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        image0 = 255 - df.loc[df.index[i]].values.reshape(137,236).astype(np.uint8)

    #normalize each image by its max val

        img = (image0*(255.0/image0.max())).astype(np.uint8)

        image = crop_resize(img)

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized



"""%%time

##Feather data generation for all train_data

for i in range(4):

    data = pd.read_parquet(PATH+f'train_image_data_{i}.parquet')

    data =Resize(data)

    data.to_feather(f'train_data_{i}{i}_l.feather')

    del data

    gc.collect()""" 

##TO save RAM I have run this command in another kernel and kept the output for the Kernel as dataset for this Kernel .
DATA_PATH = "../input/pytorch-efficientnet-starter-kernel/"
## Load Feather Data 

train_all = pd.read_csv(PATH + "train.csv")

train = train_all[train_all.grapheme_root.isin([59,60,61,62,63,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95])]

data0 = pd.read_feather(DATA_PATH+"train_data_00_l.feather")

data1 = pd.read_feather(DATA_PATH+'train_data_11_l.feather')

data2 = pd.read_feather(DATA_PATH+'train_data_22_l.feather')

data3 = pd.read_feather(DATA_PATH+'train_data_33_l.feather')

data_full1 = pd.concat([data0,data1,data2,data3],ignore_index=True)

data_full = data_full1.loc[data_full1.image_id.isin(train.image_id.values)]

del data0,data1,data2,data3,data_full1

gc.collect()

data_full.shape
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

label_grapheme = le.fit_transform(train.grapheme_root.values)

label_conso = le.fit_transform(train.consonant_diacritic.values)
train['grapheme_root'] = label_grapheme

train['consonant_diacritic'] = label_conso
## A bunch of code copied from internet . Half of them I dont understand yet . However , CutOut is used in this notebook

##https://github.com/hysts/pytorch_image_classification

import numpy as np

import torch

import torch.nn as nn



class Cutout:

    def __init__(self, mask_size, p, cutout_inside, mask_color=1):

        self.p = p

        self.mask_size = mask_size

        self.cutout_inside = cutout_inside

        self.mask_color = mask_color



        self.mask_size_half = mask_size // 2

        self.offset = 1 if mask_size % 2 == 0 else 0



    def __call__(self, image):

        image = np.asarray(image).copy()



        if np.random.random() > self.p:

            return image



        h, w = image.shape[:2]



        if self.cutout_inside:

            cxmin, cxmax = self.mask_size_half, w + self.offset - self.mask_size_half

            cymin, cymax = self.mask_size_half, h + self.offset - self.mask_size_half

        else:

            cxmin, cxmax = 0, w + self.offset

            cymin, cymax = 0, h + self.offset



        cx = np.random.randint(cxmin, cxmax)

        cy = np.random.randint(cymin, cymax)

        xmin = cx - self.mask_size_half

        ymin = cy - self.mask_size_half

        xmax = xmin + self.mask_size

        ymax = ymin + self.mask_size

        xmin = max(0, xmin)

        ymin = max(0, ymin)

        xmax = min(w, xmax)

        ymax = min(h, ymax)

        image[ymin:ymax, xmin:xmax] = self.mask_color

        return image





class DualCutout:

    def __init__(self, mask_size, p, cutout_inside, mask_color=1):

        self.cutout = Cutout(mask_size, p, cutout_inside, mask_color)



    def __call__(self, image):

        return np.hstack([self.cutout(image), self.cutout(image)])





class DualCutoutCriterion:

    def __init__(self, alpha):

        self.alpha = alpha

        self.criterion = nn.CrossEntropyLoss(reduction='mean')



    def __call__(self, preds, targets):

        preds1, preds2 = preds

        return (self.criterion(preds1, targets) + self.criterion(

            preds2, targets)) * 0.5 + self.alpha * F.mse_loss(preds1, preds2)





def mixup(data, targets, alpha, n_classes):

    indices = torch.randperm(data.size(0))

    shuffled_data = data[indices]

    shuffled_targets = targets[indices]



    lam = np.random.beta(alpha, alpha)

    data = data * lam + shuffled_data * (1 - lam)

    targets = (targets, shuffled_targets, lam)



    return data, targets





def mixup_criterion(preds, targets):

    targets1, targets2, lam = targets

    criterion = nn.CrossEntropyLoss(reduction='mean')

    return lam * criterion(preds, targets1) + (1 - lam) * criterion(

        preds, targets2)

    





class RandomErasing:

    def __init__(self, p, area_ratio_range, min_aspect_ratio, max_attempt):

        self.p = p

        self.max_attempt = max_attempt

        self.sl, self.sh = area_ratio_range

        self.rl, self.rh = min_aspect_ratio, 1. / min_aspect_ratio



    def __call__(self, image):

        image = np.asarray(image).copy()



        if np.random.random() > self.p:

            return image



        h, w = image.shape[:2]

        image_area = h * w



        for _ in range(self.max_attempt):

            mask_area = np.random.uniform(self.sl, self.sh) * image_area

            aspect_ratio = np.random.uniform(self.rl, self.rh)

            mask_h = int(np.sqrt(mask_area * aspect_ratio))

            mask_w = int(np.sqrt(mask_area / aspect_ratio))



            if mask_w < w and mask_h < h:

                x0 = np.random.randint(0, w - mask_w)

                y0 = np.random.randint(0, h - mask_h)

                x1 = x0 + mask_w

                y1 = y0 + mask_h

                image[y0:y1, x0:x1] = np.random.uniform(0, 1)

                break



        return image  
## Add Augmentations as suited from Albumentations library

train_aug = Compose([ 

    ShiftScaleRotate(p=1,border_mode=cv2.BORDER_CONSTANT,value =1),

    OneOf([

        ElasticTransform(p=0.1, alpha=1, sigma=50, alpha_affine=50,border_mode=cv2.BORDER_CONSTANT,value =1),

        GridDistortion(distort_limit =0.05 ,border_mode=cv2.BORDER_CONSTANT,value =1, p=0.1),

        OpticalDistortion(p=0.1, distort_limit= 0.05, shift_limit=0.2,border_mode=cv2.BORDER_CONSTANT,value =1)                  

        ], p=0.3),

    OneOf([

        GaussNoise(var_limit=1.0),

        Blur(),

        GaussianBlur(blur_limit=3)

        ], p=0.4),    

    RandomGamma(p=0.8)])



## A lot of heavy augmentations
## Someone asked for normalization of images . values collected from Iafoss





class ToTensor:

    def __call__(self, data):

        if isinstance(data, tuple):

            return tuple([self._to_tensor(image) for image in data])

        else:

            return self._to_tensor(data)



    def _to_tensor(self, data):

        if len(data.shape) == 3:

            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))

        else:

            return torch.from_numpy(data[None, :, :].astype(np.float32))





class Normalize:

    def __init__(self, mean, std):

        self.mean = np.array(mean)

        self.std = np.array(std)



    def __call__(self, image):

        image = np.asarray(image).astype(np.float32) / 255.

        image = (image - self.mean) / self.std

        return image
## Create dataset function

class GraphemeDataset(Dataset):

    def __init__(self,df,label,_type='train',transform =True,aug=train_aug):

        self.df = df

        self.label = label

        self.aug = aug

        self.transform = transform

        self.data = df.iloc[:, 1:].values

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):

        label1 = self.label.vowel_diacritic.values[idx]

        label2 = self.label.grapheme_root.values[idx]

        label3 = self.label.consonant_diacritic.values[idx]

        #image = self.df.iloc[idx][1:].values.reshape(128,128).astype(np.float)

        image = self.data[idx, :].reshape(128,128).astype(np.float)

        if self.transform:

            augment = self.aug(image =image)

            image = augment['image']

            cutout = Cutout(32,0.5,True,1)

            image = cutout(image)

        norm = Normalize([0.0692],[0.2051])

        image = norm(image)



        return image,label1,label2,label3
## Do a train-valid split of the data to create dataset and dataloader . Specify random seed to get reproducibility 

from sklearn.model_selection import train_test_split

train_df , valid_df = train_test_split(train,test_size=0.20, random_state=42,shuffle=True) ## Split Labels

data_train_df, data_valid_df = train_test_split(data_full,test_size=0.20, random_state=42,shuffle =True) ## split data

del data_full 

gc.collect()
##Creating the train and valid dataset for training . Training data has the transform flag ON

train_dataset = GraphemeDataset(data_train_df ,train_df,transform = True) 

valid_dataset = GraphemeDataset(data_valid_df ,valid_df,transform = False) 

torch.cuda.empty_cache()

gc.collect()
##Visulization function for checking Original and augmented image

def visualize(original_image,aug_image):

    fontsize = 18

    

    f, ax = plt.subplots(1, 2, figsize=(8, 8))



    ax[0].imshow(original_image, cmap='gray')

    ax[0].set_title('Original image', fontsize=fontsize)

    ax[1].imshow(aug_image,cmap='gray')

    ax[1].set_title('Augmented image', fontsize=fontsize)

    
## One image taken from raw dataframe another from dataset 

orig_image = data_train_df.iloc[0, 1:].values.reshape(128,128).astype(np.float)

aug_image = train_dataset[0][0]
## Check the augmentations 

for i in range (20):

    aug_image = train_dataset[0][0]

    visualize (orig_image,aug_image)
del train_df,valid_df,data_train_df,data_valid_df 

torch.cuda.empty_cache()

gc.collect()
## Create data loader and get ready for training .

batch_size = 32 

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=True)

## Mish Activation Function Not yet Used . May be later 

class Mish(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, x): 

        

        x = x *( torch.tanh(F.softplus(x)))



        return x
## Over9000 Optimizer . Inspired by Iafoss . Over and Out !

##https://github.com/mgrankin/over9000/blob/master/ralamb.py

import torch, math

from torch.optim.optimizer import Optimizer



# RAdam + LARS

class Ralamb(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.buffer = [[None, None, None] for ind in range(10)]

        super(Ralamb, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(Ralamb, self).__setstate__(state)



    def step(self, closure=None):



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('Ralamb does not support sparse gradients')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                # Decay the first and second moment running average coefficient

                # m_t

                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # v_t

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)



                state['step'] += 1

                buffered = self.buffer[int(state['step'] % 10)]



                if state['step'] == buffered[0]:

                    N_sma, radam_step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    # more conservative since it's an approximated value

                    if N_sma >= 5:

                        radam_step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    else:

                        radam_step_size = 1.0 / (1 - beta1 ** state['step'])

                    buffered[2] = radam_step_size



                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)



                # more conservative since it's an approximated value

                radam_step = p_data_fp32.clone()

                if N_sma >= 5:

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    radam_step.addcdiv_(-radam_step_size * group['lr'], exp_avg, denom)

                else:

                    radam_step.add_(-radam_step_size * group['lr'], exp_avg)



                radam_norm = radam_step.pow(2).sum().sqrt()

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                if weight_norm == 0 or radam_norm == 0:

                    trust_ratio = 1

                else:

                    trust_ratio = weight_norm / radam_norm



                state['weight_norm'] = weight_norm

                state['adam_norm'] = radam_norm

                state['trust_ratio'] = trust_ratio



                if N_sma >= 5:

                    p_data_fp32.addcdiv_(-radam_step_size * group['lr'] * trust_ratio, exp_avg, denom)

                else:

                    p_data_fp32.add_(-radam_step_size * group['lr'] * trust_ratio, exp_avg)



                p.data.copy_(p_data_fp32)



        return loss



# Lookahead implementation from https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lookahead.py



""" Lookahead Optimizer Wrapper.

Implementation modified from: https://github.com/alphadl/lookahead.pytorch

Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610

"""

import torch

from torch.optim.optimizer import Optimizer

from collections import defaultdict



class Lookahead(Optimizer):

    def __init__(self, base_optimizer, alpha=0.5, k=6):

        if not 0.0 <= alpha <= 1.0:

            raise ValueError(f'Invalid slow update rate: {alpha}')

        if not 1 <= k:

            raise ValueError(f'Invalid lookahead steps: {k}')

        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)

        self.base_optimizer = base_optimizer

        self.param_groups = self.base_optimizer.param_groups

        self.defaults = base_optimizer.defaults

        self.defaults.update(defaults)

        self.state = defaultdict(dict)

        # manually add our defaults to the param groups

        for name, default in defaults.items():

            for group in self.param_groups:

                group.setdefault(name, default)



    def update_slow(self, group):

        for fast_p in group["params"]:

            if fast_p.grad is None:

                continue

            param_state = self.state[fast_p]

            if 'slow_buffer' not in param_state:

                param_state['slow_buffer'] = torch.empty_like(fast_p.data)

                param_state['slow_buffer'].copy_(fast_p.data)

            slow = param_state['slow_buffer']

            slow.add_(group['lookahead_alpha'], fast_p.data - slow)

            fast_p.data.copy_(slow)



    def sync_lookahead(self):

        for group in self.param_groups:

            self.update_slow(group)



    def step(self, closure=None):

        # print(self.k)

        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)

        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:

            group['lookahead_step'] += 1

            if group['lookahead_step'] % group['lookahead_k'] == 0:

                self.update_slow(group)

        return loss



    def state_dict(self):

        fast_state_dict = self.base_optimizer.state_dict()

        slow_state = {

            (id(k) if isinstance(k, torch.Tensor) else k): v

            for k, v in self.state.items()

        }

        fast_state = fast_state_dict['state']

        param_groups = fast_state_dict['param_groups']

        return {

            'state': fast_state,

            'slow_state': slow_state,

            'param_groups': param_groups,

        }



    def load_state_dict(self, state_dict):

        fast_state_dict = {

            'state': state_dict['state'],

            'param_groups': state_dict['param_groups'],

        }

        self.base_optimizer.load_state_dict(fast_state_dict)



        # We want to restore the slow state, but share param_groups reference

        # with base_optimizer. This is a bit redundant but least code

        slow_state_new = False

        if 'slow_state' not in state_dict:

            print('Loading state_dict from optimizer without Lookahead applied.')

            state_dict['slow_state'] = defaultdict(dict)

            slow_state_new = True

        slow_state_dict = {

            'state': state_dict['slow_state'],

            'param_groups': state_dict['param_groups'],  # this is pointless but saves code

        }

        super(Lookahead, self).load_state_dict(slow_state_dict)

        self.param_groups = self.base_optimizer.param_groups  # make both ref same container

        if slow_state_new:

            # reapply defaults to catch missing lookahead specific ones

            for name, default in self.defaults.items():

                for group in self.param_groups:

                    group.setdefault(name, default)



def LookaheadAdam(params, alpha=0.5, k=6, *args, **kwargs):

     adam = Adam(params, *args, **kwargs)

     return Lookahead(adam, alpha, k)





# RAdam + LARS + LookAHead



# Lookahead implementation from https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py

# RAdam + LARS implementation from https://gist.github.com/redknightlois/c4023d393eb8f92bb44b2ab582d7ec20



def Over9000(params, alpha=0.5, k=6, *args, **kwargs):

     ralamb = Ralamb(params, *args, **kwargs)

     return Lookahead(ralamb, alpha, k)



RangerLars = Over9000 
## Code copied from Lukemelas github repository have a look

## https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch

"""

This file contains helper functions for building the model and for loading model parameters.

These helper functions are built to mirror those in the official TensorFlow implementation.

"""



import re

import math

import collections

from functools import partial

import torch

from torch import nn

from torch.nn import functional as F

from torch.utils import model_zoo



########################################################################

############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############

########################################################################





# Parameters for the entire model (stem, all blocks, and head)

GlobalParams = collections.namedtuple('GlobalParams', [

    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',

    'num_classes', 'width_coefficient', 'depth_coefficient',

    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])



# Parameters for an individual model block

BlockArgs = collections.namedtuple('BlockArgs', [

    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',

    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])



# Change namedtuple defaults

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)





class SwishImplementation(torch.autograd.Function):

    @staticmethod

    def forward(ctx, i):

        result = i * torch.sigmoid(i)

        ctx.save_for_backward(i)

        return result



    @staticmethod

    def backward(ctx, grad_output):

        i = ctx.saved_variables[0]

        sigmoid_i = torch.sigmoid(i)

        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))





class MemoryEfficientSwish(nn.Module):

    def forward(self, x):

        return SwishImplementation.apply(x)



class Swish(nn.Module):

    def forward(self, x):

        return x * torch.sigmoid(x)





def round_filters(filters, global_params):

    """ Calculate and round number of filters based on depth multiplier. """

    multiplier = global_params.width_coefficient

    if not multiplier:

        return filters

    divisor = global_params.depth_divisor

    min_depth = global_params.min_depth

    filters *= multiplier

    min_depth = min_depth or divisor

    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)

    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%

        new_filters += divisor

    return int(new_filters)





def round_repeats(repeats, global_params):

    """ Round number of filters based on depth multiplier. """

    multiplier = global_params.depth_coefficient

    if not multiplier:

        return repeats

    return int(math.ceil(multiplier * repeats))





def drop_connect(inputs, p, training):

    """ Drop connect. """

    if not training: return inputs

    batch_size = inputs.shape[0]

    keep_prob = 1 - p

    random_tensor = keep_prob

    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)

    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor

    return output





def get_same_padding_conv2d(image_size=None):

    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.

        Static padding is necessary for ONNX exporting of models. """

    if image_size is None:

        return Conv2dDynamicSamePadding

    else:

        return partial(Conv2dStaticSamePadding, image_size=image_size)





class Conv2dDynamicSamePadding(nn.Conv2d):

    """ 2D Convolutions like TensorFlow, for a dynamic image size """



    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):

        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2



    def forward(self, x):

        ih, iw = x.size()[-2:]

        kh, kw = self.weight.size()[-2:]

        sh, sw = self.stride

        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)

        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)

        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:

            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)





class Conv2dStaticSamePadding(nn.Conv2d):

    """ 2D Convolutions like TensorFlow, for a fixed image size"""



    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):

        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2



        # Calculate padding based on image size and save it

        assert image_size is not None

        ih, iw = image_size if type(image_size) == list else [image_size, image_size]

        kh, kw = self.weight.size()[-2:]

        sh, sw = self.stride

        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)

        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)

        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:

            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        else:

            self.static_padding = Identity()



    def forward(self, x):

        x = self.static_padding(x)

        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return x





class Identity(nn.Module):

    def __init__(self, ):

        super(Identity, self).__init__()



    def forward(self, input):

        return input





########################################################################

############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############

########################################################################





def efficientnet_params(model_name):

    """ Map EfficientNet model name to parameter coefficients. """

    params_dict = {

        # Coefficients:   width,depth,res,dropout

        'efficientnet-b0': (1.0, 1.0, 224, 0.2),

        'efficientnet-b1': (1.0, 1.1, 240, 0.2),

        'efficientnet-b2': (1.1, 1.2, 260, 0.3),

        'efficientnet-b3': (1.2, 1.4, 300, 0.3),

        'efficientnet-b4': (1.4, 1.8, 380, 0.4),

        'efficientnet-b5': (1.6, 2.2, 456, 0.4),

        'efficientnet-b6': (1.8, 2.6, 528, 0.5),

        'efficientnet-b7': (2.0, 3.1, 600, 0.5),

    }

    return params_dict[model_name]





class BlockDecoder(object):

    """ Block Decoder for readability, straight from the official TensorFlow repository """



    @staticmethod

    def _decode_block_string(block_string):

        """ Gets a block through a string notation of arguments. """

        assert isinstance(block_string, str)



        ops = block_string.split('_')

        options = {}

        for op in ops:

            splits = re.split(r'(\d.*)', op)

            if len(splits) >= 2:

                key, value = splits[:2]

                options[key] = value



        # Check stride

        assert (('s' in options and len(options['s']) == 1) or

                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))



        return BlockArgs(

            kernel_size=int(options['k']),

            num_repeat=int(options['r']),

            input_filters=int(options['i']),

            output_filters=int(options['o']),

            expand_ratio=int(options['e']),

            id_skip=('noskip' not in block_string),

            se_ratio=float(options['se']) if 'se' in options else None,

            stride=[int(options['s'][0])])



    @staticmethod

    def _encode_block_string(block):

        """Encodes a block to a string."""

        args = [

            'r%d' % block.num_repeat,

            'k%d' % block.kernel_size,

            's%d%d' % (block.strides[0], block.strides[1]),

            'e%s' % block.expand_ratio,

            'i%d' % block.input_filters,

            'o%d' % block.output_filters

        ]

        if 0 < block.se_ratio <= 1:

            args.append('se%s' % block.se_ratio)

        if block.id_skip is False:

            args.append('noskip')

        return '_'.join(args)



    @staticmethod

    def decode(string_list):

        """

        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block

        :return: a list of BlockArgs namedtuples of block args

        """

        assert isinstance(string_list, list)

        blocks_args = []

        for block_string in string_list:

            blocks_args.append(BlockDecoder._decode_block_string(block_string))

        return blocks_args



    @staticmethod

    def encode(blocks_args):

        """

        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args

        :return: a list of strings, each string is a notation of block

        """

        block_strings = []

        for block in blocks_args:

            block_strings.append(BlockDecoder._encode_block_string(block))

        return block_strings





def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,

                 drop_connect_rate=0.2, image_size=None, num_classes=1000):

    """ Creates a efficientnet model. """



    blocks_args = [

        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',

        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',

        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',

        'r1_k3_s11_e6_i192_o320_se0.25',

    ]

    blocks_args = BlockDecoder.decode(blocks_args)



    global_params = GlobalParams(

        batch_norm_momentum=0.99,

        batch_norm_epsilon=1e-3,

        dropout_rate=dropout_rate,

        drop_connect_rate=drop_connect_rate,

        # data_format='channels_last',  # removed, this is always true in PyTorch

        num_classes=num_classes,

        width_coefficient=width_coefficient,

        depth_coefficient=depth_coefficient,

        depth_divisor=8,

        min_depth=None,

        image_size=image_size,

    )



    return blocks_args, global_params





def get_model_params(model_name, override_params):

    """ Get the block args and global params for a given model """

    if model_name.startswith('efficientnet'):

        w, d, s, p = efficientnet_params(model_name)

        # note: all models have drop connect rate = 0.2

        blocks_args, global_params = efficientnet(

            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)

    else:

        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:

        # ValueError will be raised here if override_params has fields not included in global_params.

        global_params = global_params._replace(**override_params)

    return blocks_args, global_params





url_map = {

    'efficientnet-b0': '../input/efficientnet-pytorch/efficientnet-b0-08094119.pth',

    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',

    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth',

    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth',

    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth',

    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth',

    'efficientnet-b6': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth',

    'efficientnet-b7': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth',

}



## This below function is modified to use the pretrained weight for single channel . Its nothing but summing the weight across one axis .

def load_pretrained_weights(model, model_name, load_fc=True,ch=1):

    """ Loads pretrained weights, and downloads if loading for the first time. """

    state_dict = torch.load('../input/efficientnet-pytorch/efficientnet-b0-08094119.pth')

    if load_fc:

        if ch == 1:

            conv1_weight = state_dict['_conv_stem.weight']

            state_dict['_conv_stem.weight'] = conv1_weight.sum(dim=1, keepdim=True)

        model.load_state_dict(state_dict)

        

    else:

        state_dict.pop('_fc.weight')

        state_dict.pop('_fc.bias')

        if ch == 1:

            conv1_weight = state_dict['_conv_stem.weight']

            state_dict['_conv_stem.weight'] = conv1_weight.sum(dim=1, keepdim=True)

        res = model.load_state_dict(state_dict, strict=False)

        print(res.missing_keys)

        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias','fc1.weight', 'fc1.bias','fc2.weight', 'fc2.bias','fc3.weight', 'fc3.bias']), 'issue loading pretrained weights'

    print('Loaded pretrained weights for {}'.format(model_name))
import torch

from torch import nn

from torch.nn import functional as F





class MBConvBlock(nn.Module):

    """

    Mobile Inverted Residual Bottleneck Block

    Args:

        block_args (namedtuple): BlockArgs, see above

        global_params (namedtuple): GlobalParam, see above

    Attributes:

        has_se (bool): Whether the block contains a Squeeze and Excitation layer.

    """



    def __init__(self, block_args, global_params):

        super().__init__()

        self._block_args = block_args

        self._bn_mom = 1 - global_params.batch_norm_momentum

        self._bn_eps = global_params.batch_norm_epsilon

        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)

        self.id_skip = block_args.id_skip  # skip connection and drop connect



        # Get static or dynamic convolution depending on image size

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)



        # Expansion phase

        inp = self._block_args.input_filters  # number of input channels

        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels

        if self._block_args.expand_ratio != 1:

            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)

            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)



        # Depthwise convolution phase

        k = self._block_args.kernel_size

        s = self._block_args.stride

        self._depthwise_conv = Conv2d(

            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise

            kernel_size=k, stride=s, bias=False)

        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)



        # Squeeze and Excitation layer, if desired

        if self.has_se:

            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))

            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)

            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)



        # Output phase

        final_oup = self._block_args.output_filters

        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)

        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

        self._swish = MemoryEfficientSwish()



    def forward(self, inputs, drop_connect_rate=None):

        """

        :param inputs: input tensor

        :param drop_connect_rate: drop connect rate (float, between 0 and 1)

        :return: output of block

        """



        # Expansion and Depthwise Convolution

        x = inputs

        if self._block_args.expand_ratio != 1:

            x = self._swish(self._bn0(self._expand_conv(inputs)))

        x = self._swish(self._bn1(self._depthwise_conv(x)))



        # Squeeze and Excitation

        if self.has_se:

            x_squeezed = F.adaptive_avg_pool2d(x, 1)

            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))

            x = torch.sigmoid(x_squeezed) * x



        x = self._bn2(self._project_conv(x))



        # Skip connection and drop connect

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters

        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:

            if drop_connect_rate:

                x = drop_connect(x, p=drop_connect_rate, training=self.training)

            x = x + inputs  # skip connection

        return x



    def set_swish(self, memory_efficient=True):

        """Sets swish function as memory efficient (for training) or standard (for export)"""

        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()





class EfficientNet(nn.Module):

    """

    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:

        blocks_args (list): A list of BlockArgs to construct blocks

        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:

        model = EfficientNet.from_pretrained('efficientnet-b0')

    """



    def __init__(self, blocks_args=None, global_params=None):

        super().__init__()

        assert isinstance(blocks_args, list), 'blocks_args should be a list'

        assert len(blocks_args) > 0, 'block args must be greater than 0'

        self._global_params = global_params

        self._blocks_args = blocks_args



        # Get static or dynamic convolution depending on image size

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)



        # Batch norm parameters

        bn_mom = 1 - self._global_params.batch_norm_momentum

        bn_eps = self._global_params.batch_norm_epsilon



        # Stem

        in_channels = 1  # rgb

        out_channels = round_filters(32, self._global_params)  # number of output channels

        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)



        # Build blocks

        self._blocks = nn.ModuleList([])

        for block_args in self._blocks_args:



            # Update block input and output filters based on depth multiplier.

            block_args = block_args._replace(

                input_filters=round_filters(block_args.input_filters, self._global_params),

                output_filters=round_filters(block_args.output_filters, self._global_params),

                num_repeat=round_repeats(block_args.num_repeat, self._global_params)

            )



            # The first block needs to take care of stride and filter size increase.

            self._blocks.append(MBConvBlock(block_args, self._global_params))

            if block_args.num_repeat > 1:

                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)

            for _ in range(block_args.num_repeat - 1):

                self._blocks.append(MBConvBlock(block_args, self._global_params))



        # Head

        in_channels = block_args.output_filters  # output of final block

        out_channels = round_filters(1280, self._global_params)

        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)



        # Final linear layer

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self._dropout = nn.Dropout(self._global_params.dropout_rate)

        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # vowel_diacritic

        self.fc1 = nn.Linear(out_channels,11)

        # grapheme_root

        self.fc2 = nn.Linear(out_channels,20)

        # consonant_diacritic

        self.fc3 = nn.Linear(out_channels,4)

        self._swish = MemoryEfficientSwish()



    def set_swish(self, memory_efficient=True):

        """Sets swish function as memory efficient (for training) or standard (for export)"""

        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

        for block in self._blocks:

            block.set_swish(memory_efficient)





    def extract_features(self, inputs):

        """ Returns output of the final convolution layer """



        # Stem

        x = self._swish(self._bn0(self._conv_stem(inputs)))



        # Blocks

        for idx, block in enumerate(self._blocks):

            drop_connect_rate = self._global_params.drop_connect_rate

            if drop_connect_rate:

                drop_connect_rate *= float(idx) / len(self._blocks)

            x = block(x, drop_connect_rate=drop_connect_rate)



        # Head

        x = self._swish(self._bn1(self._conv_head(x)))



        return x



    def forward(self, inputs):

        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        bs = inputs.size(0)

        # Convolution layers

        x = self.extract_features(inputs)



        # Pooling and final linear layer

        x = self._avg_pooling(x)

        x = x.view(bs, -1)

        x = self._dropout(x)

       # x = self._fc(x)

        x1 = self.fc1(x)

        x2= self.fc2(x)

        x3 = self.fc3(x)

        return x1,x2,x3



    @classmethod

    def from_name(cls, model_name, override_params=None):

        cls._check_model_name_is_valid(model_name)

        blocks_args, global_params = get_model_params(model_name, override_params)

        return cls(blocks_args, global_params)



    @classmethod

    def from_pretrained(cls, model_name, num_classes=1000, in_channels = 1):

        model = cls.from_name(model_name, override_params={'num_classes': num_classes})

        load_pretrained_weights(model, model_name, load_fc=False)

        if in_channels != 3:

            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)

            out_channels = round_filters(32, model._global_params)

            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

        return model

    

    @classmethod

    def from_pretrained(cls, model_name, num_classes=1000):

        model = cls.from_name(model_name, override_params={'num_classes': num_classes})

        load_pretrained_weights(model, model_name, load_fc=False)



        return model



    @classmethod

    def get_image_size(cls, model_name):

        cls._check_model_name_is_valid(model_name)

        _, _, res, _ = efficientnet_params(model_name)

        return res



    @classmethod

    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):

        """ Validates model name. None that pretrained weights are only available for

        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """

        num_models = 4 if also_need_pretrained_weights else 8

        valid_models = ['efficientnet-b'+str(i) for i in range(num_models)]

        if model_name not in valid_models:

            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
## Make sure we are using the GPU . Get CUDA device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
## Now create the model. Since its greyscale , I have  used pretrained model with modified weight. 

##I will make the necessary modification to load pretrained weights for greyscale by summing up the weights over one axis or copying greyscale into three channels

model = EfficientNet.from_pretrained('efficientnet-b0').to(device) ## I switched to effnet-b4 in this version 
## A Small but useful test of the Model by using dummy input . .

x = torch.zeros((32,1, 64, 64))

with torch.no_grad():

    output1,output2,output3 =model(x.to(device))

print(output3.shape)
## This is a placeholder for finetunign or inference when you want to load a previously trained model

##and want to finetune or want to do just inference



##model.load_state_dict(torch.load('../input/bengef2/effnetb0_trial_stage1.pth'))  

## There is a small thing . I trained using effnetb4 offline for 20 epochs and loaded the weight

## I forgot to change the naming convension and it still reads effnetb0 . But this is actually effnetb4
n_epochs = 1 ## 1 Epoch as sample . "I am just a poor boy  , no GPU in reality "



#optimizer =torch.optim.Adam(model.parameters(), lr=1e-4)

#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, 2e-4) ## This didnt give good result need to correct  and get the right scheduler .

optimizer =Over9000(model.parameters(), lr=2e-3, weight_decay=1e-3) ## New once 

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=None, epochs=n_epochs, steps_per_epoch=5021, pct_start=0.0,

                                   anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0) ## Scheduler . Step for each batch

criterion = nn.CrossEntropyLoss()

batch_size=32
##Local Metrics implementation .

##https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch

import numpy as np

import sklearn.metrics

import torch





def macro_recall(pred_y, y, n_grapheme=20, n_vowel=11, n_consonant=4):

    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)

    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]



    y = y.cpu().numpy()

    # pred_y = [p.cpu().numpy() for p in pred_y]



    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], average='macro')

    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], average='macro')

    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], average='macro')

    scores = [recall_grapheme, recall_vowel, recall_consonant]

    final_score = np.average(scores, weights=[2, 1, 1])

    # print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, '

    #       f'total {final_score}, y {y.shape}')

    return final_score



def macro_recall_multi(pred_graphemes, true_graphemes,pred_vowels,true_vowels,pred_consonants,true_consonants, n_grapheme=20, n_vowel=11, n_consonant=4):

    #pred_y = torch.split(pred_y, [n_grapheme], dim=1)

    pred_label_graphemes = torch.argmax(pred_graphemes, dim=1).cpu().numpy()



    true_label_graphemes = true_graphemes.cpu().numpy()

    

    pred_label_vowels = torch.argmax(pred_vowels, dim=1).cpu().numpy()



    true_label_vowels = true_vowels.cpu().numpy()

    

    pred_label_consonants = torch.argmax(pred_consonants, dim=1).cpu().numpy()



    true_label_consonants = true_consonants.cpu().numpy()    

    # pred_y = [p.cpu().numpy() for p in pred_y]



    recall_grapheme = sklearn.metrics.recall_score(pred_label_graphemes, true_label_graphemes, average='macro')

    recall_vowel = sklearn.metrics.recall_score(pred_label_vowels, true_label_vowels, average='macro')

    recall_consonant = sklearn.metrics.recall_score(pred_label_consonants, true_label_consonants, average='macro')

    scores = [recall_grapheme, recall_vowel, recall_consonant]

    final_score = np.average(scores, weights=[2, 1, 1])

    #print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, '

    #       f'total {final_score}')

    return final_score





def calc_macro_recall(solution, submission):

    # solution df, submission df

    scores = []

    for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:

        y_true_subset = solution[solution[component] == component]['target'].values

        y_pred_subset = submission[submission[component] == component]['target'].values

        scores.append(sklearn.metrics.recall_score(

            y_true_subset, y_pred_subset, average='macro'))

    final_score = np.average(scores, weights=[2, 1, 1])

    return final_score
## This function for train is copied from @hanjoonchoe

## We are going to train and track accuracy and then evaluate and track validation accuracy

def train(epoch,history):

  model.train()

  losses = []

  accs = []

  acc= 0.0

  total = 0.0

  running_loss = 0.0

  running_acc = 0.0

  running_recall = 0.0

  for idx, (inputs,labels1,labels2,labels3) in tqdm(enumerate(train_loader),total=len(train_loader)):

      inputs = inputs.to(device)

      labels1 = labels1.to(device)

      labels2 = labels2.to(device)

      labels3 = labels3.to(device)

      total += len(inputs)

      optimizer.zero_grad()

      outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float())

      loss1 = 0.1*criterion(outputs1,labels1)

      loss2 = 0.7* criterion(outputs2,labels2)

      loss3 = 0.2*criterion(outputs3,labels3)

      running_loss += loss1.item()+loss2.item()+loss3.item()

      running_recall+= macro_recall_multi(outputs2,labels2,outputs1,labels1,outputs3,labels3)

      running_acc += (outputs1.argmax(1)==labels1).float().mean()

      running_acc += (outputs2.argmax(1)==labels2).float().mean()

      running_acc += (outputs3.argmax(1)==labels3).float().mean()

      (loss1+loss2+loss3).backward()

      optimizer.step()

      optimizer.zero_grad()

      acc = running_acc/total

      scheduler.step()

  losses.append(running_loss/len(train_loader))

  accs.append(running_acc/(len(train_loader)*3))

  print(' train epoch : {}\tacc : {:.2f}%'.format(epoch,running_acc/(len(train_loader)*3)))

  print('loss : {:.4f}'.format(running_loss/len(train_loader)))

    

  print('recall: {:.4f}'.format(running_recall/len(train_loader)))

  total_train_recall = running_recall/len(train_loader)

  torch.cuda.empty_cache()

  gc.collect()

  history.loc[epoch, 'train_loss'] = losses[0]

  history.loc[epoch,'train_acc'] = accs[0].cpu().numpy()

  history.loc[epoch,'train_recall'] = total_train_recall

  return  total_train_recall

def evaluate(epoch,history):

   model.eval()

   losses = []

   accs = []

   recalls = []

   acc= 0.0

   total = 0.0

   #print('epochs {}/{} '.format(epoch+1,epochs))

   running_loss = 0.0

   running_acc = 0.0

   running_recall = 0.0

   with torch.no_grad():

     for idx, (inputs,labels1,labels2,labels3) in tqdm(enumerate(valid_loader),total=len(valid_loader)):

        inputs = inputs.to(device)

        labels1 = labels1.to(device)

        labels2 = labels2.to(device)

        labels3 = labels3.to(device)

        total += len(inputs)

        outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float())

        loss1 = criterion(outputs1,labels1)

        loss2 = 2*criterion(outputs2,labels2)

        loss3 = criterion(outputs3,labels3)

        running_loss += loss1.item()+loss2.item()+loss3.item()

        running_recall+= macro_recall_multi(outputs2,labels2,outputs1,labels1,outputs3,labels3)

        running_acc += (outputs1.argmax(1)==labels1).float().mean()

        running_acc += (outputs2.argmax(1)==labels2).float().mean()

        running_acc += (outputs3.argmax(1)==labels3).float().mean()

        acc = running_acc/total

        #scheduler.step()

   losses.append(running_loss/len(valid_loader))

   accs.append(running_acc/(len(valid_loader)*3))

   recalls.append(running_recall/len(valid_loader))

   total_recall = running_recall/len(valid_loader) ## No its not Arnold Schwarzenegger movie

   print('val epoch: {} \tval acc : {:.2f}%'.format(epoch,running_acc/(len(valid_loader)*3)))

   print('loss : {:.4f}'.format(running_loss/len(valid_loader)))

   print('recall: {:.4f}'.format(running_recall/len(valid_loader)))

   history.loc[epoch, 'valid_loss'] = losses[0]

   history.loc[epoch, 'valid_acc'] = accs[0].cpu().numpy()

   history.loc[epoch, 'valid_recall'] = total_recall

   return  total_recall

## A very simple loop to train for number of epochs it probably can be made more robust to save only the file with best valid loss 

history = pd.DataFrame()

n_epochs = 30 ## 1 Epoch as sample . "I am just a poor boy  , no GPU in reality "

valid_recall = 0.0

best_valid_recall = 0.0

for epoch in range(n_epochs):

    torch.cuda.empty_cache()

    gc.collect()

    train_recall = train(epoch,history)

    valid_recall = evaluate(epoch,history)

    if valid_recall > best_valid_recall:

        print(f'Validation recall has increased from:  {best_valid_recall:.4f} to: {valid_recall:.4f}. Saving checkpoint')

        torch.save(model.state_dict(), 'effnetb0_trial_stage1.pth') ## Saving model weights based on best validation accuracy.

        best_valid_recall = valid_recall ## Set the new validation Recall score to compare with next epoch

        

        

    #scheduler.step() ## Want to test with fixed learning rate .If you want to use scheduler please uncomment this .

    
history.to_csv('history.csv')

history.head()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import cv2

import torch

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms,models

from tqdm import tqdm_notebook as tqdm
## Load model for inferernce . 

#model.load_state_dict(torch.load('../input/pytorch-efficientnet-starter-code/effnetb0_trial_stage1.pth')) 
test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
#check https://www.kaggle.com/iafoss/image-preprocessing-128x128



def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=SIZE, pad=16):

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    #cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    #remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    #make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(img,(size,size))
class GraphemeDataset(Dataset):

    def __init__(self, fname):

        print(fname)

        self.df = pd.read_parquet(fname)

        self.data = 255 - self.df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        name = self.df.iloc[idx,0]

        #normalize each image by its max val

        img = (self.data[idx]*(255.0/self.data[idx].max())).astype(np.uint8)

        img = crop_resize(img)

        img = img.astype(np.float32)/255.0

        return img, name
## All test data

test_data = ['/kaggle/input/bengaliai-cv19/test_image_data_0.parquet','/kaggle/input/bengaliai-cv19/test_image_data_1.parquet','/kaggle/input/bengaliai-cv19/test_image_data_2.parquet',

             '/kaggle/input/bengaliai-cv19/test_image_data_3.parquet']
%%time

## Inference a little faster using @Iafoss and  @peters technique

row_id,target = [],[]

for fname in test_data:

    #data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/{fname}')

    test_image = GraphemeDataset(fname)

    dl = torch.utils.data.DataLoader(test_image,batch_size=128,num_workers=4,shuffle=False)

    with torch.no_grad():

        for x,y in tqdm(dl):

            x = x.unsqueeze(1).float().cuda()

            p1,p2,p3 = model(x)

            p1 = p1.argmax(-1).view(-1).cpu()

            p2 = p2.argmax(-1).view(-1).cpu()

            p3 = p3.argmax(-1).view(-1).cpu()

            for idx,name in enumerate(y):

                row_id += [f'{name}_vowel_diacritic',f'{name}_grapheme_root',

                           f'{name}_consonant_diacritic']

                target += [p1[idx].item(),p2[idx].item(),p3[idx].item()]

                

sub_df = pd.DataFrame({'row_id': row_id, 'target': target})

sub_df.to_csv('submission.csv', index=False)

sub_df.head(20)