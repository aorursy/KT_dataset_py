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



WEIGHTS_FILE = '/kaggle/input/mobilenet-v2-30-epochs/model_2020-03-03T12_16_37.627015.pt'





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
"""

Copy entirely torchvision.models.mobilenet_v2 simply to force myself to read through mobilenet V2

"""



import torch

from torch import nn

#from yacs.config import CfgNode

#from .build import BACKBONE_REGISTRY





def _make_divisible(v, divisor, min_value=None):

    """

    This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8

    It can be seen here:

    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    :param v:

    :param divisor:

    :param min_value:

    :return:

    """

    if min_value is None:

        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.

    if new_v < 0.9 * v:

        new_v += divisor

    return new_v





class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):

        padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__(

            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),

            nn.BatchNorm2d(out_planes),

            nn.ReLU6(inplace=True)

        )





class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):

        super(InvertedResidual, self).__init__()

        self.stride = stride

        assert stride in [1, 2]



        hidden_dim = int(round(inp * expand_ratio))

        self.use_res_connect = self.stride == 1 and inp == oup



        layers = []

        if expand_ratio != 1:

            # pw

            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([

            # dw

            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),

            # pw-linear

            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),

            nn.BatchNorm2d(oup),

        ])

        self.conv = nn.Sequential(*layers)



    def forward(self, x):

        if self.use_res_connect:

            return x + self.conv(x)

        else:

            return self.conv(x)





class MobileNetV2(nn.Module):

    def __init__(self,

                 width_mult=1.0,

                 inverted_residual_setting=None,

                 round_nearest=8,

                 block=None):

        """

        MobileNet V2 main class



        """

        super(MobileNetV2, self).__init__()



        if block is None:

            block = InvertedResidual

        input_channel = 32

        last_channel = 1280



        if inverted_residual_setting is None:

            inverted_residual_setting = [

                # t, c, n, s

                [1, 16, 1, 1],

                [6, 24, 2, 2],

                [6, 32, 3, 2],

                [6, 64, 4, 2],

                [6, 96, 3, 1],

                [6, 160, 3, 2],

                [6, 320, 1, 1],

            ]



        # only check the first element, assuming user knows t,c,n,s are required

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:

            raise ValueError("inverted_residual_setting should be non-empty "

                             "or a 4-element list, got {}".format(inverted_residual_setting))



        # building first layer

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(3, input_channel, stride=2)]

        # building inverted residual blocks

        for t, c, n, s in inverted_residual_setting:

            output_channel = _make_divisible(c * width_mult, round_nearest)

            for i in range(n):

                stride = s if i == 0 else 1

                features.append(block(input_channel, output_channel, stride, expand_ratio=t))

                input_channel = output_channel

        # building last several layers

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        # make it nn.Sequential

        self.features = nn.Sequential(*features)



        # weight initialization

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out')

                if m.bias is not None:

                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.ones_(m.weight)

                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, 0, 0.01)

                nn.init.zeros_(m.bias)



    def _forward_impl(self, x):

        # This exists since TorchScript doesn't support inheritance, so the superclass method

        # (this one) needs to have a name other than `forward` that can be accessed in a subclass

        x = self.features(x)

        x = x.mean([2, 3])

        return x



    def forward(self, x):

        return self._forward_impl(x)
def build_mobilenet_backbone(backbone_cfg, **kwargs):

    """

    :param backbone_cfg: backbone config node

    :param kwargs:

    :return: backbone module

    """

    model = MobileNetV2(**kwargs)

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

                'input_dims': 1280,   # densenet121

                'hidden_dims': [512, 256],

                'bn': True,

                'dropout': -1

                        },

            'backbone_cfg': {

#                 'pretrained_path': '/kaggle/input/julien-4-epochs-densenet121-bengali/model.pt'

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

        self.backbone = build_mobilenet_backbone(model_cfg['backbone_cfg'])

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