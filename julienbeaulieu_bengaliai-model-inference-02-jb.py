import pickle

import gc

import os

import cv2 

from cv2 import resize

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from pathlib import Path



import torch

from torch import nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



from typing import Union

from typing import List
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
DEFAULT_H, DEFAULT_W = 137, 236



SIZE = 128



LABEL_PATH = Path('train.csv')



DATADIR = Path('/kaggle/input/bengaliai-cv19')



TEST_FORM = 'test_image_data_ID.parquet'



WEIGHTS_FILE = '/kaggle/input/densenet121-80-epochs-ohem-07/model_2020-03-07T12_40_50.436927.pt'



test = pd.read_csv(DATADIR/'test.csv')

train = pd.read_csv(DATADIR/'train.csv')

train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

submission_df = pd.read_csv(DATADIR/'sample_submission.csv')
# loading PARQUET format files 

def load_images(train_test, indices=['0', '1', '2', '3']):

    """

    Utility function to Load the images from both the location and return them

    :param train_test:

    :return:

    """



    path_form = {

        'test': TEST_FORM

    }[train_test]



    imgs_list = []



    # sequentially load all four files.

    for id in indices:



        # Form the path of the files.

        path = DATADIR / path_form.replace('ID', id)

        print('Loading', path)

        df = pd.read_parquet(path)

        imgs = df.iloc[:, 1:].to_numpy()

        imgs_list.append(imgs)

    del imgs

    gc.collect()

    imgs_list = np.concatenate(imgs_list)

    imgs_list = imgs_list.reshape(-1, DEFAULT_H, DEFAULT_W)



    return imgs_list
def get_data(train_test, indices=['0', '1', '2', '3']):

    """

     A combined function to load both trian and label?

    :return:

    """

    # Load all images into a variable.

    imgs = load_images(train_test, indices=indices)

    

    if train_test == 'train':

        labels = load_labels()

        all_data = list(zip(imgs, labels))

    else:

        all_data = imgs



    return all_data
# use a dictionary as config settings

dataset_cfg = {'aug_cfg': {

                    'resize_shape': (128, 128),

                    'crop': True,

                    'to_rgb': True,

                    'normalize_mean': [0.485, 0.456, 0.406],

                    'normalize_std': [0.229, 0.224, 0.225]

                          }

              }
def content_crop(img, pad_to_square: bool, white_background: bool):

    """

    https://www.kaggle.com/iafoss/image-preprocessing-128x128



    :param img: grapheme image matrix

    :param pad_to_square:  whether pad to square (preserving aspect ratio)

    :param white_background: whether the image

    :return: cropped image matrix

    """

    # remove the surrounding 5 pixels

    img = img[5:-5, 5:-5]

    if white_background:

        y_list, x_list = np.where(img < 235)

    else:

        y_list, x_list = np.where(img > 80)



    # get xy min max

    xmin, xmax = np.min(x_list), np.max(x_list)

    ymin, ymax = np.min(y_list), np.max(y_list)



    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < 223) else 236

    ymax = ymax + 10 if (ymax < 127) else 137

    img = img[ymin:ymax, xmin:xmax]



    # remove lo intensity pixels as noise

    if white_background:

        img[img > 235] = 255

    else:

        img[img < 28] = 0

        

    if pad_to_square:

        lx, ly = xmax - xmin, ymax - ymin

        l = max(lx, ly) + 16

        # make sure that the aspect ratio is kept in rescaling

        if white_background:

            constant_pad = 255

        else:

            constant_pad = 0

        img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode='constant', constant_values=constant_pad)



    return img



class Preprocessor(object):

    

    def __init__(self, dataset_cfg):

        aug_cfg = dataset_cfg['aug_cfg']

        self.resize_shape = aug_cfg['resize_shape']

        self.crop = aug_cfg['crop']

        self.to_rgb = aug_cfg['to_rgb']

        self.normalize_mean = aug_cfg['normalize_mean']

        self.normalize_std = aug_cfg['normalize_std']

                                     

    def __call__(self, img, normalize=True):

            

        if self.crop:

            img = content_crop(img, pad_to_square=True, white_background=True)

        

        img = resize(img, self.resize_shape)

        

        if self.to_rgb: 

            img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)

        

        if not normalize:

            return img

        

        # normalize to 0-1

        img = img / 255.

        

        if self.normalize_mean is not None:

            img = (img - self.normalize_mean) / self.normalize_std

       

        img = torch.tensor(img)

        img = img.permute([2, 0, 1])



        return img
# return an image and the name of the image

class BengaliDataset(Dataset):

    """

    Torch data set object for the bengali data

    """



    def __init__(self, data_list, data_cfg, fname, indices=None):

        """

        :param data_list: list of raw data consists of (image, labels)

        :param data_cfg:  data config node

        """

        self.data_list = data_list

        self.data_size = len(data_list)



        if indices is None:

            indices = np.arange(self.data_size)

        self.indices = indices

        self.preprocessor = Preprocessor(data_cfg)

        

        # get image names

        if fname:

            self.df = pd.read_parquet(DATADIR / fname)

        self.fname = fname



    def __len__(self) -> int:

        return len(self.indices)



    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):

        idx = self.indices[idx]      

        img = self.data_list[idx]

        img = self.preprocessor(img)

        name = self.df.iloc[idx, 0]

        return img, name
# # Use collator if batch size > 1

# class BengaliDataBatchCollator(object):

#     """

#     Custom collator

#     """



#     def __init__(self):

#         pass

    

#     def __call__(self, batch: List) -> (torch.Tensor, torch.Tensor):

#         """

#         :param batch:

#         :return:

#         """



#         inputs = np.array([x[0] for x in batch])

#         inputs = torch.tensor(inputs)

#         inputs = inputs.permute([0, 3, 1, 2])

#         names = [x[1] for x in batch]

#         return inputs, names
# densenet121 - copy pasted from Pytorch github

import re

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.utils.checkpoint as cp

from collections import OrderedDict

from torch import Tensor

from torch.jit.annotations import List



from torch.hub import load_state_dict_from_url





__all__ = ['DenseNet', 'densenet121']



model_urls = {

    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',

}



class _DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):

        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),

        self.add_module('relu1', nn.ReLU(inplace=True)),

        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *

                                           growth_rate, kernel_size=1, stride=1,

                                           bias=False)),

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),

        self.add_module('relu2', nn.ReLU(inplace=True)),

        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,

                                           kernel_size=3, stride=1, padding=1,

                                           bias=False)),

        self.drop_rate = float(drop_rate)

        self.memory_efficient = memory_efficient



    def bn_function(self, inputs):

        # type: (List[Tensor]) -> Tensor

        concated_features = torch.cat(inputs, 1)

        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484

        return bottleneck_output



    # todo: rewrite when torchscript supports any

    def any_requires_grad(self, input):

        # type: (List[Tensor]) -> bool

        for tensor in input:

            if tensor.requires_grad:

                return True

        return False



    @torch.jit.unused  # noqa: T484

    def call_checkpoint_bottleneck(self, input):

        # type: (List[Tensor]) -> Tensor

        def closure(*inputs):

            return self.bn_function(*inputs)



        return cp.checkpoint(closure, input)



    @torch.jit._overload_method  # noqa: F811

    def forward(self, input):

        # type: (List[Tensor]) -> (Tensor)

        pass



    @torch.jit._overload_method  # noqa: F811

    def forward(self, input):

        # type: (Tensor) -> (Tensor)

        pass



    # torchscript does not yet support *args, so we overload method

    # allowing it to take either a List[Tensor] or single Tensor

    def forward(self, input):  # noqa: F811

        if isinstance(input, Tensor):

            prev_features = [input]

        else:

            prev_features = input



        if self.memory_efficient and self.any_requires_grad(prev_features):

            if torch.jit.is_scripting():

                raise Exception("Memory Efficient not supported in JIT")



            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)

        else:

            bottleneck_output = self.bn_function(prev_features)



        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:

            new_features = F.dropout(new_features, p=self.drop_rate,

                                     training=self.training)

        return new_features





class _DenseBlock(nn.ModuleDict):

    _version = 2



    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):

        super(_DenseBlock, self).__init__()

        for i in range(num_layers):

            layer = _DenseLayer(

                num_input_features + i * growth_rate,

                growth_rate=growth_rate,

                bn_size=bn_size,

                drop_rate=drop_rate,

                memory_efficient=memory_efficient,

            )

            self.add_module('denselayer%d' % (i + 1), layer)



    def forward(self, init_features):

        features = [init_features]

        for name, layer in self.items():

            new_features = layer(features)

            features.append(new_features)

        return torch.cat(features, 1)





class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):

        super(_Transition, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(num_input_features))

        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,

                                          kernel_size=1, stride=1, bias=False))

        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))





class DenseNet(nn.Module):

    r"""Densenet-BC model class, based on

    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:

        growth_rate (int) - how many filters to add each layer (`k` in paper)

        block_config (list of 4 ints) - how many layers in each pooling block

        num_init_features (int) - the number of filters to learn in the first convolution layer

        bn_size (int) - multiplicative factor for number of bottle neck layers

          (i.e. bn_size * k features in the bottleneck layer)

        drop_rate (float) - dropout rate after each dense layer

        num_classes (int) - number of classification classes

        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,

          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_

    """



    __constants__ = ['features']



    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),

                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):



        super(DenseNet, self).__init__()



        # First convolution

        self.features = nn.Sequential(OrderedDict([

            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,

                                padding=3, bias=False)),

            ('norm0', nn.BatchNorm2d(num_init_features)),

            ('relu0', nn.ReLU(inplace=True)),

            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ]))



        # Each denseblock

        num_features = num_init_features

        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(

                num_layers=num_layers,

                num_input_features=num_features,

                bn_size=bn_size,

                growth_rate=growth_rate,

                drop_rate=drop_rate,

                memory_efficient=memory_efficient

            )

            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:

                trans = _Transition(num_input_features=num_features,

                                    num_output_features=num_features // 2)

                self.features.add_module('transition%d' % (i + 1), trans)

                num_features = num_features // 2



        # Final batch norm

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))



        # Linear layer

        self.classifier = nn.Linear(num_features, num_classes)



        # Official init from torch repo.

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.constant_(m.bias, 0)



    def forward(self, x):

        features = self.features(x)

        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = torch.flatten(out, 1)

        out = self.classifier(out)

        return out





def _load_state_dict(model, model_url, progress):

    # '.'s are no longer allowed in module names, but previous _DenseLayer

    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.

    # They are also in the checkpoints in model_urls. This pattern is used

    # to find such keys.

    pattern = re.compile(

        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')



    state_dict = load_state_dict_from_url(model_url, progress=progress)

    for key in list(state_dict.keys()):

        res = pattern.match(key)

        if res:

            new_key = res.group(1) + res.group(2)

            state_dict[new_key] = state_dict[key]

            del state_dict[key]

    model.load_state_dict(state_dict)





def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,

              **kwargs):

    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)

    if pretrained:

        _load_state_dict(model, model_urls[arch], progress)

    return model





def densenet121(pretrained=False, progress=True, **kwargs):

    r"""Densenet-121 model from

    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,

          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_

    """

    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,

                     **kwargs)
def build_densenet_backbone(backbone_cfg, **kwargs):

    """

    :param backbone_cfg: backbone config node

    :param kwargs:

    :return: backbone module

    """

    model = densenet121(pretrained=False)

    if backbone_cfg.get('pretrained_path'):

        pretrained_path = backbone_cfg['pretrained_path']

        state_dict = torch.load(pretrained_path, map_location='cpu')

        model.load_state_dict(state_dict, strict=False)

    return model
# head and backbone config

model_cfg = {

            'head_cfg': {

                'head_name': 'simple_head',

                'activation': 'leaky_relu',

                'output_dims': [168, 11, 7],

                'input_dims': 1000,   # densenet121

                'hidden_dims': [512, 256],

                'bn': True,

                'dropout': -1

                        },

            'backbone_cfg': {

                #'pretrained_path': '/kaggle/input/julien-4-epochs-densenet121-bengali/model.pt'

                            }

             }
from torch import nn

import torch.nn.functional as F

from typing import Union



ACTIVATION_FN = {

    'relu': F.relu,

    'relu6': F.relu6,

    'elu': F.elu,

    'leaky_relu': F.leaky_relu,

    None: None

}



class LinearLayer(nn.Module):



    def __init__(self, input_dim, output_dim, activation, bn, dropout_rate = -1):

        super(LinearLayer, self).__init__()

        self.input_dim = input_dim

        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)

        self.activation_fn = ACTIVATION_FN[activation]

        if bn:

            self.bn = nn.BatchNorm1d(self.output_dim)

        else:

            self.bn = None

        if dropout_rate > 0:

            self.dropout = nn.Dropout(p=dropout_rate)

        else:

            self.dropout = None



    def forward(self, x):

        # LINEAR -> BN -> ACTIVATION -> DROPOUT

        x = self.linear(x)

        if self.bn is not None:

            x = self.bn(x)

        if self.activation_fn is not None:

            x = self.activation_fn(x, inplace=True)

        if self.dropout is not None:

            x = self.dropout(x)

        return x

from torch import nn



def build_head(head_cfg):

    return SimplePredictionHead(head_cfg)





class SimplePredictionHead(nn.Module):



    def __init__(self, head_cfg):

        super(SimplePredictionHead, self).__init__()

        self.fc_layers = []

        input_dim = head_cfg['input_dims']

        # first hidden layers

        for hidden_dim in head_cfg['hidden_dims']:

            self.fc_layers.append(

                LinearLayer(input_dim, hidden_dim, bn=head_cfg['bn'], activation=head_cfg['activation'],

                            dropout_rate=head_cfg['dropout'])

            )

            input_dim = hidden_dim



        output_dims = head_cfg['output_dims']



        # prediction layer

        self.fc_layers.append(

            LinearLayer(input_dim, sum(output_dims), bn=False, activation=None, dropout_rate=-1)

        )



        self.fc_layers = nn.Sequential(*self.fc_layers)

        for m in self.modules():

            if isinstance(m, nn.BatchNorm1d):

                nn.init.ones_(m.weight)

                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, 0, 0.01)

                nn.init.zeros_(m.bias)



    def forward(self, x):



        return self.fc_layers(x)
class BaselineModel(nn.Module):



    def __init__(self, model_cfg):

        super(BaselineModel, self).__init__()

        self.backbone = build_densenet_backbone(model_cfg['backbone_cfg'])

        self.head = build_head(model_cfg['head_cfg'])

        self.heads_dims = model_cfg['head_cfg']['output_dims']



    def forward(self, x):

        x = self.backbone(x)

        x = self.head(x)

        grapheme_logits, vowel_logits, consonant_logits = torch.split(x, self.heads_dims, dim=1)

        return grapheme_logits, vowel_logits, consonant_logits
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = "cpu"
model = BaselineModel(model_cfg)

state_dict = torch.load(WEIGHTS_FILE, map_location='cpu')

model.load_state_dict(state_dict['model_state'])

model.to(device)
def test_eval():    

    model.eval()

    test_data = ['test_image_data_0.parquet','test_image_data_1.parquet','test_image_data_2.parquet','test_image_data_3.parquet']

    row_id,target = [],[]

    batch_size=1

    for idx, fname in enumerate(test_data):

        test_images = get_data('test', indices=[str(idx)])

        test_dataset = BengaliDataset(test_images, dataset_cfg, fname=fname)

        

        # test_collator = BengaliDataBatchCollator() ---> don't need batch collator for batch size of 1

        

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  

                             num_workers=4)



        with torch.no_grad():

            for inputs, name in test_loader:

                inputs = inputs.to(device)

                name = str(name).strip("'(),'")

                grapheme_logits, vowel_logits, consonant_logits = model(inputs.float())



                grapheme_logits  = grapheme_logits.argmax(-1)

                vowel_logits     = vowel_logits.argmax(-1)

                consonant_logits = consonant_logits.argmax(-1)

                

                # use a for loop if batch_size > 1

                row_id += [f'{name}_grapheme_root',f'{name}_vowel_diacritic',

                               f'{name}_consonant_diacritic']

                target += [grapheme_logits.item(), vowel_logits.item(), 

                           consonant_logits.item()]

            del test_images, test_dataset, test_loader

            gc.collect()



    return pd.DataFrame({'row_id': row_id, 'target': target})

submission_df = test_eval()
submission_df
submission_df.to_csv('submission.csv', index=False)