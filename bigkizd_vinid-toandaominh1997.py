import os

import gzip

from PIL import Image

import numpy as np 

from torch.utils.data import Dataset

import torch.optim as optim

import torchvision.models as models

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

from torch.nn import functional as F

import re

import math

import collections

from functools import partial

import torch

from torch import nn

from torch.utils import model_zoo

from torch.nn import CrossEntropyLoss

from torchvision import transforms

import cv2

import copy

from PIL import Image

import numpy as np 

import random

from imgaug import augmenters as iaa

from tqdm import tqdm_notebook as tqdm

def read_labels_from_file(filename):

    with open(filename,'rb') as f: #use gzip to open the file in read binary mode

        magic = f.read(4) # magic number is the first 4 bytes

        magic = int.from_bytes(magic,'big') # Convert bytes to integers.



        nolab = f.read(4)

        nolab = int.from_bytes(nolab,'big')

        labels = [f.read(1) for i in range(nolab)]

        labels = [int.from_bytes(label, 'big') for label in labels]

    return labels





def read_images_from_file(filename):

    with open(filename,'rb') as f:

        magic = f.read(4)

        magic = int.from_bytes(magic,'big')

        noimg = f.read(4)

        noimg = int.from_bytes(noimg,'big')

        norow = f.read(4)

        norow = int.from_bytes(norow,'big')

        nocol = f.read(4)

        nocol = int.from_bytes(nocol,'big')

        images = [] # create array

        for i in range(noimg):

            rows = []

            for r in range(norow):

                cols = []

                for c in range(nocol):

                    cols.append(int.from_bytes(f.read(1), 'big')) # append the current byte for every column

                rows.append(cols) # append columns array for every row

            images.append(rows) # append rows for every image

    return images

train_images = read_images_from_file("../input/mnistdataset/train-images-idx3-ubyte/train-images-idx3-ubyte")

test_images = read_images_from_file("../input/mnistdataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")



train_labels = read_labels_from_file("../input/mnistdataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte")

test_labels = read_labels_from_file("../input/mnistdataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")
# Transform

def random_dilate(img):

    img = np.array(img)

    img = cv2.dilate(img, np.ones(shape=(random.randint(1,3), random.randint(1,3)), dtype=np.uint8))

    return Image.fromarray(img)



def random_erode(img):

    img = np.array(img)

    img = cv2.erode(img, np.ones(shape=(random.randint(1,3), random.randint(1,3)), dtype=np.uint8))

    return Image.fromarray(img)



class ResizeImage:

    def __init__(self, height):

        self.height = height 

    def __call__(self, img):

        img = np.array(img)

        h,w = img.shape[:2]



        new_w = int(self.height / h * w)

        img = cv2.resize(img, (new_w, self.height), interpolation=cv2.INTER_CUBIC)



        return Image.fromarray(img)



class ImgAugTransform:

    def __init__(self):

        self.aug = iaa.Sequential([

            iaa.Sometimes(0.5,

                          iaa.OneOf(

                              [

                                  iaa.GaussianBlur(sigma=(0, 3.0)),

                                  iaa.AverageBlur(k=(3, 11)),

                                  iaa.MedianBlur(k=(3, 11))

                              ])

                          ),



        ])



    def __call__(self, img):

        img = np.array(img)

        transformed_img =  self.aug.augment_image(img)



        return Image.fromarray(transformed_img)

def train_transforms():

    transform = transforms.Compose([



            transforms.RandomApply(

                [

                    random_dilate,

                ],

                p=0.15),



            transforms.RandomApply(

                [

                    random_erode,

                ],

                p=0.15),



            transforms.RandomApply(

                [

                    ImgAugTransform(),

                ],

                p=0.15),

            transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=5, resample=Image.NEAREST, fillcolor=255),

            transforms.ToTensor()

        ])

    return transform

def test_transforms():

    transform = transforms.Compose([

        transforms.ToTensor() 

    ])

    return transform
class mnistDataset(Dataset):

    def __init__(self, data_images, data_labels, transform=None):

        super(mnistDataset, self).__init__()

        self.data_images = data_images

        self.data_labels = data_labels

        self.transform = transform

    def __len__(self):

        return len(self.data_images)

    def __getitem__(self, index):

        try:

            image = Image.fromarray(np.array(self.data_images[index]).astype('uint8'))

        except:

            print('Corrupted image for %d' % index)

            return self[index + 1]

        if self.transform is not None:

            image = self.transform(image)

        label = self.data_labels[index]

        return (image, label)

batch_size = 128

num_class = 10

train_transform = train_transforms()

test_transform = test_transforms()

train_dataset = mnistDataset(data_images= train_images, data_labels=train_labels, transform=train_transform)

test_dataset = mnistDataset(data_images= test_images, data_labels=test_labels, transform=test_transform)



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(8))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=int(8))    
class ConvNet(nn.Module):

    def __init__(self, num_classes=10):

        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7*7*32, num_classes)

        self.logsoft = nn.LogSoftmax(dim=1)

        

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)

        out = self.logsoft(out)

        return out

GlobalParams = collections.namedtuple('GlobalParams', [

    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',

    'num_classes', 'width_coefficient', 'depth_coefficient',

    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])







BlockArgs = collections.namedtuple('BlockArgs', [

    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',

    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])





GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)





def relu_fn(x):

    return x * torch.sigmoid(x)





def round_filters(filters, global_params):

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

    multiplier = global_params.depth_coefficient

    if not multiplier:

        return repeats

    return int(math.ceil(multiplier * repeats))





def drop_connect(inputs, p, training):

    if not training: return inputs

    batch_size = inputs.shape[0]

    keep_prob = 1 - p

    random_tensor = keep_prob

    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)

    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor

    return output





def get_same_padding_conv2d(image_size=None):

    if image_size is None:

        return Conv2dDynamicSamePadding

    else:

        return partial(Conv2dStaticSamePadding, image_size=image_size)



class Conv2dDynamicSamePadding(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):

        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2



    def forward(self, x):

        ih, iw = x.size()[-2:]

        kh, kw = self.weight.size()[-2:]

        sh, sw = self.stride

        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)

        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)

        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:

            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)





class Conv2dStaticSamePadding(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):

        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2



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

    def __init__(self,):

        super(Identity, self).__init__()



    def forward(self, input):

        return input





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

    @staticmethod

    def _decode_block_string(block_string):

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

        assert isinstance(string_list, list)

        blocks_args = []

        for block_string in string_list:

            blocks_args.append(BlockDecoder._decode_block_string(block_string))

        return blocks_args



    @staticmethod

    def encode(blocks_args):

        block_strings = []

        for block in blocks_args:

            block_strings.append(BlockDecoder._encode_block_string(block))

        return block_strings





def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,

                 drop_connect_rate=0.2, image_size=None, num_classes=1000):

    """ Creates a efficientnet model. """



    blocks_args = [

        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s11_e6_i16_o24_se0.25',

        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s11_e6_i40_o80_se0.25',

        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s2_e6_i112_o192_se0.25',

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

    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth',

    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',

    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth',

    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth',

    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth',

    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth',

    'efficientnet-b6': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth',

    'efficientnet-b7': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth',

}



def load_pretrained_weights(model, model_name, load_fc=True):

    """ Loads pretrained weights, and downloads if loading for the first time. """

    state_dict = model_zoo.load_url(url_map[model_name])

    if load_fc:

        model.load_state_dict(state_dict)

    else:

        state_dict.pop('_fc.weight')

        state_dict.pop('_fc.bias')

        res = model.load_state_dict(state_dict, strict=False)

        assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'

    print('Loaded pretrained weights for {}'.format(model_name))    

    

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



    def forward(self, inputs, drop_connect_rate=None):

        """

        :param inputs: input tensor

        :param drop_connect_rate: drop connect rate (float, between 0 and 1)

        :return: output of block

        """



        # Expansion and Depthwise Convolution

        x = inputs

        if self._block_args.expand_ratio != 1:

            x = relu_fn(self._bn0(self._expand_conv(inputs)))

        x = relu_fn(self._bn1(self._depthwise_conv(x)))



        # Squeeze and Excitation

        if self.has_se:

            x_squeezed = F.adaptive_avg_pool2d(x, 1)

            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))

            x = torch.sigmoid(x_squeezed) * x



        x = self._bn2(self._project_conv(x))



        # Skip connection and drop connect

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters

        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:

            if drop_connect_rate:

                x = drop_connect(x, p=drop_connect_rate, training=self.training)

            x = x + inputs  # skip connection

        return x





class EfficientNet(nn.Module):

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

        in_channels = 1 # rgb

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

        self._dropout = self._global_params.dropout_rate

        self._fc = nn.Linear(out_channels, self._global_params.num_classes)



    def extract_features(self, inputs):

        """ Returns output of the final convolution layer """



        # Stem

        x = relu_fn(self._bn0(self._conv_stem(inputs)))



        # Blocks

        for idx, block in enumerate(self._blocks):

            drop_connect_rate = self._global_params.drop_connect_rate

            if drop_connect_rate:

                drop_connect_rate *= float(idx) / len(self._blocks)

            x = block(x, drop_connect_rate=drop_connect_rate)



        # Head

        x = relu_fn(self._bn1(self._conv_head(x)))



        return x



    def forward(self, inputs):

        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """



        # Convolution layers

        x = self.extract_features(inputs)

        # Pooling and final linear layer

        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        if self._dropout:

            x = F.dropout(x, p=self._dropout, training=self.training)

        x = self._fc(x)

        return x



    @classmethod

    def from_name(cls, model_name, override_params=None):

        cls._check_model_name_is_valid(model_name)

        blocks_args, global_params = get_model_params(model_name, override_params)

        return EfficientNet(blocks_args, global_params)



    @classmethod

    def from_pretrained(cls, model_name, num_classes=1000):

        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})

#         load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))

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

        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]

        if model_name.replace('-','_') not in valid_models:

            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=10)

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

model = model.to(device)

criterion = CrossEntropyLoss()



parameters = copy.deepcopy(model.state_dict())

# inputs = torch.randn(4, 1, 28, 28)

# inputs = inputs.to(device)

# output = model(inputs)

# print(output.size())
def init_normal(m):

    if type(m) == nn.Linear:

        torch.nn.init.uniform_(m.weight)



def train(data_loader, criterion, optimizer):

    total_loss=0

    model.train()

#     data_loader = tqdm(data_loader)

    for idx, (images, labels) in enumerate(data_loader):

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss +=loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

    return total_loss/len(data_loader)

def train_style(data_loader, criterion, optimizer):

    total_loss=0

    model.train()

#     data_loader = tqdm(data_loader)

    for idx, (images, labels) in enumerate(data_loader):

        images = images.to(device)

        labels = labels.to(device)

        targets = torch.zeros(labels.size(0), num_class)

        for index, label in enumerate(labels):

            targets[index][label] = 1.0

        targets = targets.to(device)

        del labels

        outputs = model(images)

        loss = criterion(outputs, targets)

        total_loss +=loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

    return total_loss/len(data_loader)

def evaluate(data_loader, criterion):

    model.eval()

    total_loss=0

    correct = 0

    total = 0

    with torch.no_grad():

        for idx, (images, labels) in enumerate(data_loader):

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            

            total_loss +=loss

        total_loss /= len(data_loader)

#         print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

        return total_loss, 100*correct/total

def evaluate_style(data_loader, criterion):

    model.eval()

    total_loss=0

    correct = 0

    total = 0

    with torch.no_grad():

        for idx, (images, labels) in enumerate(data_loader):

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            targets = torch.zeros(labels.size(0), num_class)

            for index, label in enumerate(labels):

                targets[index][label] = 1.0

            targets = targets.to(device)

            labels = targets

            del targets

            loss = criterion(outputs, labels)

#             _, predicted = torch.max(outputs.data, 1)

            predicted = copy.deepcopy(labels)

            for index in range(outputs.size(0)):

                best = 0.0

                best_idx = 0

                for idx in range(outputs.size(1)):

                    if outputs[index][idx]>best:

                        best = outputs[index][idx]

                        best_idx = idx

                predicted[index] = best_idx

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            

            total_loss +=loss

        total_loss /= len(data_loader)

#         print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

        return total_loss, 100*correct/total

def init_weight(model, parameters):

    for param, _param in zip(model.parameters(), parameters):

        param = _param

def train_with_augmentation(num_epoch):

    train_transform = train_transforms()

    test_transform = test_transforms()

    train_dataset = mnistDataset(data_images= train_images, data_labels=train_labels, transform=train_transform)

    test_dataset = mnistDataset(data_images= test_images, data_labels=test_labels, transform=test_transform)



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(8))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=int(8))

    best_acc = 0

    for epoch in tqdm(range(1, num_epoch)):

        loss = train(train_loader, criterion=criterion, optimizer=optimizer)

#         print('training at epoch {}: {}'.format(epoch, loss))

        _, acc = evaluate(test_loader, criterion=criterion)

        if best_acc < acc:

            best_acc = acc

    print('Best accuracy: {}'.format(best_acc))

def train_non_with_augmentation(num_epoch):

    train_transform = transforms.Compose([

        transforms.ToTensor()

    ])

    test_transform = transforms.Compose([

        transforms.ToTensor()

    ])

    train_dataset = mnistDataset(data_images= train_images, data_labels=train_labels, transform=train_transform)

    test_dataset = mnistDataset(data_images= test_images, data_labels=test_labels, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(8))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=int(8))

    best_acc = 0

    for epoch in tqdm(range(1, num_epoch)):

        loss = train(train_loader, criterion=criterion, optimizer=optimizer)

#         print('training at epoch {}: {}'.format(epoch, loss))

        _, acc = evaluate(test_loader, criterion=criterion)

        if best_acc < acc:

            best_acc = acc

    print('Best accuracy: {}'.format(best_acc))

def task2():

    num_epoch = 30

    init_weight(model, parameters)

    print('Training with augmentation')

    train_with_augmentation(num_epoch)

#     model.apply(init_normal)

    init_weight(model, parameters)

    print('Training non-with augmentation')

    train_non_with_augmentation(num_epoch)

task2()
def __compare_optimizer(): 

    optim_adam = optim.Adam(model.parameters(), lr=0.001)

    optim_adamw = optim.AdamW(model.parameters(), lr=0.001)

    optim_adadelta = optim.Adadelta(model.parameters(), lr=0.001)

    optim_adagrad = optim.Adagrad(model.parameters(), lr=0.001)

    optim_sparseadam = optim.SparseAdam(model.parameters(), lr=0.001)

    optim_asgd = optim.ASGD(model.parameters(), lr=0.001)

    optim_lbfgs = optim.LBFGS(model.parameters(), lr=0.001)

    optim_rmsprop = optim.RMSprop(model.parameters(), lr=0.001)

    optim_rprop = optim.Rprop(model.parameters(), lr=0.001)

    optim_sgd = optim.SGD(model.parameters(), lr=0.001)

    optim_list = [

        ['Adam', optim_adam],

#         ['AdamW', optim_adamw],

#     ]

#         ['Adadelta', optim_adadelta],

        ['Adagrad', optim_adagrad],

#         ['SparseAdam', optim_sparseadam],

        ['Asgd', optim_asgd],

#         ['Lbfgs', optim_lbfgs],

        ['Rmsprop', optim_rmsprop],

#         ['Rprop', optim_rprop],

        ['SGD', optim_sgd]

    ]

    df = pd.DataFrame()

    total_loss = list()

    total_name = list()

    total_epoch = list()

    total_acc = list()

    for idx, opt in enumerate(optim_list):

        optimizer = opt[1]

        num_epoch = 10

#         model.apply(init_normal)

        init_weight(model, parameters)

        for index, epoch in enumerate(tqdm(range(num_epoch))):

            loss = train(train_loader, criterion=criterion, optimizer=optimizer)

            _, acc = evaluate(test_loader, criterion=criterion)

            total_loss.append(loss.item())

            total_name.append(opt[0])

            total_epoch.append(epoch) 

            total_acc.append(acc)

    df['Optimizer'] = total_name

    df['Loss'] = total_loss

    df['Epoch'] = total_epoch

    df['Val_accuracy'] = total_acc

    sns.lineplot(x='Epoch', y='Loss', hue='Optimizer', data=df)

    plt.show()

    sns.lineplot(x='Epoch', y='Val_accuracy', hue='Optimizer', data=df)

    plt.show()

    

    

__compare_optimizer()

    

    
def __compare_loss():

    loss_list = [

        ['MSELoss', nn.MSELoss()],

        ['CrossEntropyLoss', nn.CrossEntropyLoss()],

        ['PoissonNLLLoss', nn.PoissonNLLLoss()]

    ]

    df = pd.DataFrame()

    total_loss = list()

    total_name = list()

    total_epoch = list()

    total_acc = list()

    for idx, los in enumerate(loss_list):

        criterion = los[1]

        num_epoch = 10

#         model.apply(init_normal)

        init_weight(model, parameters)

        for index, epoch in enumerate(tqdm(range(num_epoch))):

            if los[0] in ['MSELoss', 'PoissonNLLLoss']:

                loss = train_style(train_loader, criterion=criterion, optimizer=optimizer)

                _, acc = evaluate_style(test_loader, criterion=criterion)

            else:

                loss = train(train_loader, criterion=criterion, optimizer=optimizer)

                _, acc = evaluate(test_loader, criterion=criterion)

            total_loss.append(loss.item())

            total_name.append(los[0])

            total_epoch.append(epoch)

            total_acc.append(acc)

    df['Val_accuracy'] = total_acc

    df['Criterion'] = total_name

    df['Loss'] = total_loss

    df['Epoch'] = total_epoch

    sns.lineplot(x='Epoch', y='Loss', hue='Criterion', data=df)

    plt.show()

    sns.lineplot(x='Epoch', y='Val_accuracy', hue='Criterion', data=df)

    plt.show()

    

__compare_loss()