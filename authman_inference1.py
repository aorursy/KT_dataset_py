!pip install -q catalyst > /dev/null

!pip install -q lycon > /dev/null



!git clone https://github.com/NVIDIA/apex

!pip install -v --no-cache-dir --global-option="--cuda_ext" ./apex/



!free -g

!nvidia-smi
from glob import glob

from sklearn.model_selection import GroupKFold

import cv2, os, time, random, warnings, cv2, gc, sklearn

from skimage import io

import torch

from torch import nn

from datetime import datetime

import pandas as pd

import numpy as np

import albumentations as A

import matplotlib.pyplot as plt

from albumentations.pytorch.transforms import ToTensorV2

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SequentialSampler, RandomSampler

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from catalyst.data.sampler import BalanceClassSampler



from torch import nn

from torch.nn import functional as F

from apex import amp



import re

import math

import collections

from functools import partial

import torch

from torch import nn

from torch.nn import functional as F



from tqdm.auto import tqdm

# from apex import amp

# import jpegio as jio

from PIL import Image 

import lycon

import pickle 



SEED = 42

EPS = 1e-8

REBUILD_16X16_DCT_SUMS = False

REBUILD_16X16_PXL_SUMS = False

REBUILD_JPEG_CACHE = False



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)
DATA_ROOT_PATH = '../input/alaska2-image-steganalysis'

def get_valid_transforms():

    return A.Compose([

#             A.Resize(height=512, width=512, p=1.0),

            A.Normalize(always_apply=True), # ImageNet

            ToTensorV2(p=1.0),

        ], p=1.0)
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

        'efficientnet-b8': (2.2, 3.6, 672, 0.5),

        'efficientnet-l2': (4.3, 5.3, 800, 0.5),

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





# train with Standard methods

# check more details in paper(EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks)

url_map = {

    'efficientnet-b0': '../input/noisy-student-efficientnet-b0.pth',

    'efficientnet-b1': '../input/noisy-student-efficientnet-b1.pth',

    'efficientnet-b2': '../input/noisy-student-efficientnet-b2.pth',

    'efficientnet-b3': '../input/noisy-student-efficientnet-b3.pth',

    'efficientnet-b4': '../input/noisy-student-efficientnet-b4.pth',

    'efficientnet-b5': '../input/noisy-student-efficientnet-b5.pth',

    'efficientnet-b6': '../input/noisy-student-efficientnet-b6.pth',

    'efficientnet-b7': '../input/noisy-student-efficientnet-b7.pth',

}





url_map_advprop = {

    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',

    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',

    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',

    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth',

    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth',

    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth',

    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth',

    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth',

    'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth',

}





def load_pretrained_weights(model, model_name, load_fc=True, advprop=False):

    """ Loads pretrained weights, and downloads if loading for the first time. """

    # AutoAugment or Advprop (different preprocessing)

    url_map_ = url_map_advprop if advprop else url_map

    state_dict = torch.load(url_map_[model_name])

    if load_fc:

        model.load_state_dict(state_dict)

    else:       

        state_dict.pop('_fc.weight')

        state_dict.pop('_fc.bias')

        res = model.load_state_dict(state_dict, strict=False)



    print(res)

    print('Loaded pretrained weights for {}'.format(model_name))

def DCT_Basis():

    N = 8

    basis = []

    for u in range(8):

        for v in range(8):

            z = np.zeros((N,N))

            for i in range(N):

                for j in range(N):

                    z[i,j] = np.cos(np.pi*(2*i+1)*u / (2*N)) * np.cos(np.pi*(2*j+1)*v / (2*N))

            basis.append(z)



    return torch.Tensor([basis,basis,basis]).permute((1,0,2,3))



def gem(x, p=3, eps=1e-6):

    x = x.double() # x=x.to(torch.float32) # comment this during inference

    x = F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    return x.half() # Comment this line in inference code use ## return x 



class GeM(nn.Module):

    # [Half Precision GeM](https://www.kaggle.com/c/bengaliai-cv19/discussion/128911):

    def __init__(self, p=3, eps=1e-6):

        super(GeM,self).__init__()

        self.p = nn.Parameter(torch.ones(1)*p)

        self.eps = eps



    def forward(self, x):

        return gem(x, self.p, self.eps)

        

    def __repr__(self):

        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'



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



    def set_trainable(self, trainable):

        self._expand_conv.requires_grad_(trainable)

        self._bn0.requires_grad_(trainable)

        self._depthwise_conv.requires_grad_(trainable)

        self._bn1.requires_grad_(trainable)

        self._se_reduce.requires_grad_(trainable)

        self._se_expand.requires_grad_(trainable)        

        self._project_conv.requires_grad_(trainable)

        self._bn2.requires_grad_(trainable)



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

        in_channels = 3  # rgb

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

        #self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self._pxls = nn.Conv2d(out_channels, 1, 1)

        

        #self._gem = GeM()

        self._dropout = nn.Dropout(self._global_params.dropout_rate)

        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        self._fc_aux = nn.Linear(out_channels, 2)

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

        x = self.extract_features(inputs)



        # At this point, the iamge is 16x16

        pxls = self._pxls(x) # No dropout

        

        #x = self._gem(x)

        x = F.adaptive_avg_pool2d(x, 1)

        

        x = x.view(bs, -1)

        x = self._dropout(x)

        

        aux = self._fc_aux(x)

        x = self._fc(x)

        return x, aux, pxls



    @classmethod

    def from_name(cls, model_name, override_params=None):

        cls._check_model_name_is_valid(model_name)

        blocks_args, global_params = get_model_params(model_name, override_params)

        return cls(blocks_args, global_params)



    @classmethod

    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3):

        model = cls.from_name(model_name, override_params={'num_classes': num_classes})

        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)

        if in_channels != 3:

            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)

            out_channels = round_filters(32, model._global_params)

            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

        return model

    

    @classmethod

    def get_image_size(cls, model_name):

        cls._check_model_name_is_valid(model_name)

        _, _, res, _ = efficientnet_params(model_name)

        return res



    @classmethod

    def _check_model_name_is_valid(cls, model_name):

        """ Validates model name. """ 

        valid_models = ['efficientnet-b'+str(i) for i in range(9)]

        if model_name not in valid_models:

            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

# net = EfficientNet.from_pretrained('efficientnet-b3', num_classes=4)

net = EfficientNet.from_name('efficientnet-b3', override_params={'num_classes':4})

net
# RELOAD BEST MODEL

checkpoint = torch.load('../input/alaska25/best-checkpoint-040epoch.bin')['model_state_dict']

for key in list(checkpoint.keys()):

    nkey = key[len('module.'):]

    checkpoint[nkey] = checkpoint[key]

    del checkpoint[key]
net.load_state_dict(checkpoint)

net.eval()



net = net.cuda()

net = amp.initialize(net, opt_level='O1')

net = torch.nn.DataParallel(net, device_ids=[0])
def get_test_transforms(mode):

    if mode == 0:

        return A.Compose([

            #A.Resize(height=512, width=512, p=1.0),

            A.Normalize(always_apply=True), # ImageNet

            ToTensorV2(p=1.0),

        ], p=1.0)

    

    elif mode == 1:

        return A.Compose([

            A.HorizontalFlip(p=1),

            #A.Resize(height=512, width=512, p=1.0),

            A.Normalize(always_apply=True), # ImageNet

            ToTensorV2(p=1.0),

        ], p=1.0)

    

    elif mode == 2:

        return A.Compose([

            A.VerticalFlip(p=1),

            #A.Resize(height=512, width=512, p=1.0),

            A.Normalize(always_apply=True), # ImageNet

            ToTensorV2(p=1.0),

        ], p=1.0)

    

    elif mode == 3:

        return A.Compose([

            A.InvertImg(p=1),

            #A.Resize(height=512, width=512, p=1.0),

            A.Normalize(always_apply=True), # ImageNet

            ToTensorV2(p=1.0),

        ], p=1.0)



    else:

        return A.Compose([

            A.HorizontalFlip(p=1),

            A.VerticalFlip(p=1),

            #A.Resize(height=512, width=512, p=1.0),

            A.Normalize(always_apply=True), # ImageNet

            ToTensorV2(p=1.0),

        ], p=1.0)
class DatasetSubmissionRetriever(Dataset):

    def __init__(self, kinds, image_names, transforms=None):

        super().__init__()

        self.kinds = kinds

        self.image_names = image_names

        self.transforms = transforms



    def __getitem__(self, index: int):

        image_name = self.image_names[index]

        image_kind = self.kinds[index]

        

        image = lycon.load(f'{DATA_ROOT_PATH}/{image_kind}/{image_name}') # Already RGB

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']



        return image_name, image



    def __len__(self) -> int:

        return self.image_names.shape[0]
# del net

del data_loader

import gc

del images

gc.collect()

torch.cuda.empty_cache()

!nvidia-smi
torch.cuda.empty_cache()
results = []

test_imgs = glob('../input/alaska2-image-steganalysis/Test/*.jpg')

for mode in range(0, 5):

    dataset = DatasetSubmissionRetriever(

        image_names=np.array([path.split('/')[-1] for path in test_imgs]),

        kinds=['Test']*len(test_imgs),

        transforms=get_test_transforms(mode),

    )



    data_loader = DataLoader(

        dataset,

        batch_size=26,

        shuffle=False,

        num_workers=4,

        drop_last=False,

    )

    

    result = {'Id': [], 'Label': []}

    for step, (image_names, images) in enumerate(data_loader):

        torch.cuda.empty_cache()

        print(step, end='\r')



        y_pred = net(images.cuda())[0] # regular outputs are the first output

        y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]



        result['Id'].extend(image_names)

        result['Label'].extend(y_pred)

        

    results.append(result)

    print('done with mode', mode)
submissions = []

for mode in range(0,5):

    submission = pd.DataFrame(results[mode])

    submissions.append(submission)

    submissions[mode].to_csv(f'submission_{mode}.csv', index=False)
# submissions[0]['Label'] = (submissions[0]['Label']*3 + submissions[1]['Label'] + submissions[2]['Label'] + submissions[3]['Label']) / 6



submissions[0]['Label'] = (

    submissions[0]['Label'] /3 +

    submissions[1]['Label'] /6 +

    submissions[2]['Label'] /6 +

    submissions[4]['Label'] /6 +

    submissions[3]['Label'] /12 # inverted image...

)





submissions[0].to_csv(f'submission.csv', index=False)

submissions[0]['Label'].hist(bins=100)

plt.show()

submissions[0].head()