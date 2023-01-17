!pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null # no output
import gc

import os

from pathlib import Path

import random

import sys



from tqdm.notebook import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- setup ---

pd.set_option('max_columns', 50)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
debug=True

submission=False

batch_size=32

device='cuda:0'

out='.'

image_size=64

arch='pretrained'

model_name='se_resnext50_32x4d'
datadir = Path('/kaggle/input/bengaliai-cv19')

featherdir = Path('/kaggle/input/bengaliaicv19feather')

outdir = Path('.')
# Read in the data CSV files

# train = pd.read_csv(datadir/'train.csv')

# test = pd.read_csv(datadir/'test.csv')

# sample_submission = pd.read_csv(datadir/'sample_submission.csv')

# class_map = pd.read_csv(datadir/'class_map.csv')
import numpy as np

import pandas as pd

import gc





def prepare_image(datadir, featherdir, data_type='train',

                  submission=False, indices=[0, 1, 2, 3]):

    assert data_type in ['train', 'test']

    if submission:

        image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet')

                         for i in indices]

    else:

        image_df_list = [pd.read_feather(featherdir / f'{data_type}_image_data_{i}.feather')

                         for i in indices]



    print('image_df_list', len(image_df_list))

    HEIGHT = 137

    WIDTH = 236

    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]

    del image_df_list

    gc.collect()

    images = np.concatenate(images, axis=0)

    return images

%%time



train = pd.read_csv(datadir/'train.csv')

train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

indices = [0] if debug else [0, 1, 2, 3]

train_images = prepare_image(

    datadir, featherdir, data_type='train', submission=False, indices=indices)

"""

Referenced `chainer.dataset.DatasetMixin` to work with pytorch Dataset.

"""

import numpy

import six

import torch

from torch.utils.data.dataset import Dataset





class DatasetMixin(Dataset):



    def __init__(self, transform=None):

        self.transform = transform



    def __getitem__(self, index):

        """Returns an example or a sequence of examples."""

        if torch.is_tensor(index):

            index = index.tolist()

        if isinstance(index, slice):

            current, stop, step = index.indices(len(self))

            return [self.get_example_wrapper(i) for i in

                    six.moves.range(current, stop, step)]

        elif isinstance(index, list) or isinstance(index, numpy.ndarray):

            return [self.get_example_wrapper(i) for i in index]

        else:

            return self.get_example_wrapper(index)



    def __len__(self):

        """Returns the number of data points."""

        raise NotImplementedError



    def get_example_wrapper(self, i):

        """Wrapper of `get_example`, to apply `transform` if necessary"""

        example = self.get_example(i)

        if self.transform:

            example = self.transform(example)

        return example



    def get_example(self, i):

        """Returns the i-th example.



        Implementations should override it. It should raise :class:`IndexError`

        if the index is invalid.



        Args:

            i (int): The index of the example.



        Returns:

            The i-th example.



        """

        raise NotImplementedError

import numpy as np





class BengaliAIDataset(DatasetMixin):

    def __init__(self, images, labels=None, transform=None, indices=None):

        super(BengaliAIDataset, self).__init__(transform=transform)

        self.images = images

        self.labels = labels

        if indices is None:

            indices = np.arange(len(images))

        self.indices = indices

        self.train = labels is not None



    def __len__(self):

        """return length of this dataset"""

        return len(self.indices)



    def get_example(self, i):

        """Return i-th data"""

        i = self.indices[i]

        x = self.images[i]

        # Opposite white and black: background will be white and

        # for future Affine transformation

        x = (255 - x).astype(np.float32) / 255.

        if self.train:

            y = self.labels[i]

            return x, y

        else:

            return x

train_dataset = BengaliAIDataset(train_images, train_labels)
image, label = train_dataset[0]

print('image', image.shape, 'label', label)
nrow, ncol = 5, 6



fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))

axes = axes.flatten()

for i, ax in tqdm(enumerate(axes)):

    image, label = train_dataset[i]

    ax.imshow(image, cmap='Greys')

    ax.set_title(f'label: {label}')

plt.tight_layout()

plt.show()

plt.savefig('bengaliai.png')
"""

From https://www.kaggle.com/corochann/deep-learning-cnn-with-chainer-lb-0-99700

"""

import cv2

from skimage.transform import AffineTransform, warp

import numpy as np





def affine_image(img):

    """



    Args:

        img: (h, w) or (1, h, w)



    Returns:

        img: (h, w)

    """

    # ch, h, w = img.shape

    # img = img / 255.

    if img.ndim == 3:

        img = img[0]



    # --- scale ---

    min_scale = 0.8

    max_scale = 1.2

    sx = np.random.uniform(min_scale, max_scale)

    sy = np.random.uniform(min_scale, max_scale)



    # --- rotation ---

    max_rot_angle = 7

    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.



    # --- shear ---

    max_shear_angle = 10

    shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.



    # --- translation ---

    max_translation = 4

    tx = np.random.randint(-max_translation, max_translation)

    ty = np.random.randint(-max_translation, max_translation)



    tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,

                            translation=(tx, ty))

    transformed_image = warp(img, tform)

    assert transformed_image.ndim == 2

    return transformed_image





def crop_char_image(image, threshold=5./255.):

    assert image.ndim == 2

    is_black = image > threshold



    is_black_vertical = np.sum(is_black, axis=0) > 0

    is_black_horizontal = np.sum(is_black, axis=1) > 0

    left = np.argmax(is_black_horizontal)

    right = np.argmax(is_black_horizontal[::-1])

    top = np.argmax(is_black_vertical)

    bottom = np.argmax(is_black_vertical[::-1])

    height, width = image.shape

    cropped_image = image[left:height - right, top:width - bottom]

    return cropped_image





def resize(image, size=(128, 128)):

    return cv2.resize(image, size)
nrow, ncol = 1, 6



fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))

axes = axes.flatten()

for i, ax in tqdm(enumerate(axes)):

    image, label = train_dataset[0]

    ax.imshow(affine_image(image), cmap='Greys')

    ax.set_title(f'label: {label}')

plt.tight_layout()
nrow, ncol = 5, 6



fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))

axes = axes.flatten()

for i, ax in tqdm(enumerate(axes)):

    image, label = train_dataset[i]

    ax.imshow(crop_char_image(image, threshold=20./255.), cmap='Greys')

    ax.set_title(f'label: {label}')

plt.tight_layout()
nrow, ncol = 5, 6



fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))

axes = axes.flatten()

for i, ax in tqdm(enumerate(axes)):

    image, label = train_dataset[i]

    ax.imshow(resize(crop_char_image(image, threshold=20./255.)), cmap='Greys')

    ax.set_title(f'label: {label}')

plt.tight_layout()
import albumentations as A

import numpy as np





def add_gaussian_noise(x, sigma):

    x += np.random.randn(*x.shape) * sigma

    x = np.clip(x, 0., 1.)

    return x





def _evaluate_ratio(ratio):

    if ratio <= 0.:

        return False

    return np.random.uniform() < ratio





def apply_aug(aug, image):

    return aug(image=image)['image']





class Transform:

    def __init__(self, affine=True, crop=True, size=(64, 64),

                 normalize=True, train=True, threshold=40.,

                 sigma=-1., blur_ratio=0., noise_ratio=0., cutout_ratio=0.,

                 grid_distortion_ratio=0., elastic_distortion_ratio=0., random_brightness_ratio=0.,

                 piece_affine_ratio=0., ssr_ratio=0.):

        self.affine = affine

        self.crop = crop

        self.size = size

        self.normalize = normalize

        self.train = train

        self.threshold = threshold / 255.

        self.sigma = sigma / 255.



        self.blur_ratio = blur_ratio

        self.noise_ratio = noise_ratio

        self.cutout_ratio = cutout_ratio

        self.grid_distortion_ratio = grid_distortion_ratio

        self.elastic_distortion_ratio = elastic_distortion_ratio

        self.random_brightness_ratio = random_brightness_ratio

        self.piece_affine_ratio = piece_affine_ratio

        self.ssr_ratio = ssr_ratio



    def __call__(self, example):

        if self.train:

            x, y = example

        else:

            x = example

        # --- Augmentation ---

        if self.affine:

            x = affine_image(x)



        # --- Train/Test common preprocessing ---

        if self.crop:

            x = crop_char_image(x, threshold=self.threshold)

        if self.size is not None:

            x = resize(x, size=self.size)

        if self.sigma > 0.:

            x = add_gaussian_noise(x, sigma=self.sigma)



        # albumentations...

        x = x.astype(np.float32)

        assert x.ndim == 2

        # 1. blur

        if _evaluate_ratio(self.blur_ratio):

            r = np.random.uniform()

            if r < 0.25:

                x = apply_aug(A.Blur(p=1.0), x)

            elif r < 0.5:

                x = apply_aug(A.MedianBlur(blur_limit=5, p=1.0), x)

            elif r < 0.75:

                x = apply_aug(A.GaussianBlur(p=1.0), x)

            else:

                x = apply_aug(A.MotionBlur(p=1.0), x)



        if _evaluate_ratio(self.noise_ratio):

            r = np.random.uniform()

            if r < 0.50:

                x = apply_aug(A.GaussNoise(var_limit=5. / 255., p=1.0), x)

            else:

                x = apply_aug(A.MultiplicativeNoise(p=1.0), x)



        if _evaluate_ratio(self.cutout_ratio):

            # A.Cutout(num_holes=2,  max_h_size=2, max_w_size=2, p=1.0)  # Deprecated...

            x = apply_aug(A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=1.0), x)



        if _evaluate_ratio(self.grid_distortion_ratio):

            x = apply_aug(A.GridDistortion(p=1.0), x)



        if _evaluate_ratio(self.elastic_distortion_ratio):

            x = apply_aug(A.ElasticTransform(

                sigma=50, alpha=1, alpha_affine=10, p=1.0), x)



        if _evaluate_ratio(self.random_brightness_ratio):

            # A.RandomBrightness(p=1.0)  # Deprecated...

            # A.RandomContrast(p=1.0)    # Deprecated...

            x = apply_aug(A.RandomBrightnessContrast(p=1.0), x)



        if _evaluate_ratio(self.piece_affine_ratio):

            x = apply_aug(A.IAAPiecewiseAffine(p=1.0), x)



        if _evaluate_ratio(self.ssr_ratio):

            x = apply_aug(A.ShiftScaleRotate(

                shift_limit=0.0625,

                scale_limit=0.1,

                rotate_limit=30,

                p=1.0), x)



        if self.normalize:

            x = (x.astype(np.float32) - 0.0692) / 0.2051

        if x.ndim == 2:

            x = x[None, :, :]

        x = x.astype(np.float32)

        if self.train:

            y = y.astype(np.int64)

            return x, y

        else:

            return x
train_transform = Transform(

    size=(image_size, image_size), threshold=20.,

    sigma=-1., blur_ratio=0.2, noise_ratio=0.2, cutout_ratio=0.2,

    grid_distortion_ratio=0.2, random_brightness_ratio=0.2,

    piece_affine_ratio=0.2, ssr_ratio=0.2)

# transform = Transform(size=(image_size, image_size)

train_dataset = BengaliAIDataset(train_images, train_labels,

                                 transform=train_transform)
nrow, ncol = 1, 6



fig, axes = plt.subplots(nrow, ncol, figsize=(20, 2))

axes = axes.flatten()

for i, ax in tqdm(enumerate(axes)):

    image, label = train_dataset[0]

    ax.imshow(image[0], cmap='Greys')

    ax.set_title(f'label: {label}')

plt.tight_layout()
nrow, ncol = 5, 6



fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))

axes = axes.flatten()

for i, ax in tqdm(enumerate(axes)):

    image, label = train_dataset[i]

    ax.imshow(image[0], cmap='Greys')

    ax.set_title(f'label: {label}')

plt.tight_layout()
import torch





def residual_add(lhs, rhs):

    lhs_ch, rhs_ch = lhs.shape[1], rhs.shape[1]

    if lhs_ch < rhs_ch:

        out = lhs + rhs[:, :lhs_ch]

    elif lhs_ch > rhs_ch:

        out = torch.cat([lhs[:, :rhs_ch] + rhs, lhs[:, rhs_ch:]], dim=1)

    else:

        out = lhs + rhs

    return out

from typing import List



import torch

from torch import nn

from torch.nn.parameter import Parameter





class LazyLoadModule(nn.Module):

    """Lazy buffer/parameter loading using load_state_dict_pre_hook



    Define all buffer/parameter in `_lazy_buffer_keys`/`_lazy_parameter_keys` and

    save buffer with `register_buffer`/`register_parameter`

    method, which can be outside of __init__ method.

    Then this module can load any shape of Tensor during de-serializing.



    Note that default value of lazy buffer is torch.Tensor([]), while lazy parameter is None.

    """

    _lazy_buffer_keys: List[str] = []     # It needs to be override to register lazy buffer

    _lazy_parameter_keys: List[str] = []  # It needs to be override to register lazy parameter



    def __init__(self):

        super(LazyLoadModule, self).__init__()

        for k in self._lazy_buffer_keys:

            self.register_buffer(k, torch.tensor([]))

        for k in self._lazy_parameter_keys:

            self.register_parameter(k, None)

        self._register_load_state_dict_pre_hook(self._hook)



    def _hook(self, state_dict, prefix, local_metadata, strict, missing_keys,

             unexpected_keys, error_msgs):

        for key in self._lazy_buffer_keys:

            self.register_buffer(key, state_dict[prefix + key])



        for key in self._lazy_parameter_keys:

            self.register_parameter(key, Parameter(state_dict[prefix + key]))

import math

import torch

from torch.nn import init

from torch.nn.parameter import Parameter

import torch.nn.functional as F





class LazyLinear(LazyLoadModule):

    """Linear module with lazy input inference



    `in_features` can be `None`, and it is determined at the first time of forward step dynamically.

    """



    __constants__ = ['bias', 'in_features', 'out_features']

    _lazy_parameter_keys = ['weight']



    def __init__(self, in_features, out_features, bias=True):

        super(LazyLinear, self).__init__()

        self.in_features = in_features

        self.out_features = out_features

        if bias:

            self.bias = Parameter(torch.Tensor(out_features))

        else:

            self.register_parameter('bias', None)



        if in_features is not None:

            self.weight = Parameter(torch.Tensor(out_features, in_features))

            self.reset_parameters()



    def reset_parameters(self):

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)

            bound = 1 / math.sqrt(fan_in)

            init.uniform_(self.bias, -bound, bound)



    def forward(self, input):

        if self.weight is None:

            self.in_features = input.shape[-1]

            self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))

            self.reset_parameters()



            # Need to send lazy defined parameter to device...

            self.to(input.device)

        return F.linear(input, self.weight, self.bias)



    def extra_repr(self):

        return 'in_features={}, out_features={}, bias={}'.format(

            self.in_features, self.out_features, self.bias is not None

        )

from torch import nn

import torch.nn.functional as F





class LinearBlock(nn.Module):



    def __init__(self, in_features, out_features, bias=True,

                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False,):

        super(LinearBlock, self).__init__()

        if in_features is None:

            self.linear = LazyLinear(in_features, out_features, bias=bias)

        else:

            self.linear = nn.Linear(in_features, out_features, bias=bias)

        if use_bn:

            self.bn = nn.BatchNorm1d(out_features)

        if dropout_ratio > 0.:

            self.dropout = nn.Dropout(p=dropout_ratio)

        else:

            self.dropout = None

        self.activation = activation

        self.use_bn = use_bn

        self.dropout_ratio = dropout_ratio

        self.residual = residual



    def __call__(self, x):

        h = self.linear(x)

        if self.use_bn:

            h = self.bn(h)

        if self.activation is not None:

            h = self.activation(h)

        if self.residual:

            h = residual_add(h, x)

        if self.dropout_ratio > 0:

            h = self.dropout(h)

        return h
import pretrainedmodels

import torch

from torch import nn

import torch.nn.functional as F

from torch.nn import Sequential





class PretrainedCNN(nn.Module):

    def __init__(self, model_name='se_resnext101_32x4d',

                 in_channels=1, out_dim=10, use_bn=True,

                 pretrained='imagenet'):

        super(PretrainedCNN, self).__init__()

        self.conv0 = nn.Conv2d(

            in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)

        activation = F.leaky_relu

        self.do_pooling = True

        if self.do_pooling:

            inch = self.base_model.last_linear.in_features

        else:

            inch = None

        hdim = 512

        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)

        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)

        self.lin_layers = Sequential(lin1, lin2)



    def forward(self, x):

        h = self.conv0(x)

        h = self.base_model.features(h)



        if self.do_pooling:

            h = torch.sum(h, dim=(-1, -2))

        else:

            # [128, 2048, 4, 4] when input is (128, 128)

            bs, ch, height, width = h.shape

            h = h.view(bs, ch*height*width)

        for layer in self.lin_layers:

            h = layer(h)

        return h
import torch

from torch import nn

import torch.nn.functional as F

from tqdm import tqdm





def accuracy(y, t):

    pred_label = torch.argmax(y, dim=1)

    count = pred_label.shape[0]

    correct = (pred_label == t).sum().type(torch.float32)

    acc = correct / count

    return acc





class BengaliClassifier(nn.Module):

    def __init__(self, predictor, n_grapheme=168, n_vowel=11, n_consonant=7):

        super(BengaliClassifier, self).__init__()

        self.n_grapheme = n_grapheme

        self.n_vowel = n_vowel

        self.n_consonant = n_consonant

        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant

        self.predictor = predictor



        self.metrics_keys = [

            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',

            'acc_grapheme', 'acc_vowel', 'acc_consonant']



    def forward(self, x, y=None):

        pred = self.predictor(x)

        if isinstance(pred, tuple):

            assert len(pred) == 3

            preds = pred

        else:

            assert pred.shape[1] == self.n_total_class

            preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)

        loss_grapheme = F.cross_entropy(preds[0], y[:, 0])

        loss_vowel = F.cross_entropy(preds[1], y[:, 1])

        loss_consonant = F.cross_entropy(preds[2], y[:, 2])

        loss = loss_grapheme + loss_vowel + loss_consonant

        metrics = {

            'loss': loss.item(),

            'loss_grapheme': loss_grapheme.item(),

            'loss_vowel': loss_vowel.item(),

            'loss_consonant': loss_consonant.item(),

            'acc_grapheme': accuracy(preds[0], y[:, 0]),

            'acc_vowel': accuracy(preds[1], y[:, 1]),

            'acc_consonant': accuracy(preds[2], y[:, 2]),

        }

        return loss, metrics, pred



    def calc(self, data_loader):

        device: torch.device = next(self.parameters()).device

        self.eval()

        output_list = []

        with torch.no_grad():

            for batch in tqdm(data_loader):

                # TODO: support general preprocessing.

                # If `data` is not `Data` instance, `to` method is not supported!

                batch = batch.to(device)

                pred = self.predictor(batch)

                output_list.append(pred)

        output = torch.cat(output_list, dim=0)

        preds = torch.split(output, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)

        return preds



    def predict_proba(self, data_loader):

        preds = self.calc(data_loader)

        return [F.softmax(p, dim=1) for p in preds]



    def predict(self, data_loader):

        preds = self.calc(data_loader)

        pred_labels = [torch.argmax(p, dim=1) for p in preds]

        return pred_labels

n_dataset = len(train_images)

train_data_size = 200 if debug else int(n_dataset * 0.9)

valid_data_size = 100 if debug else int(n_dataset - train_data_size)



perm = np.random.RandomState(777).permutation(n_dataset)

print('perm', perm)

train_dataset = BengaliAIDataset(

    train_images, train_labels, transform=Transform(size=(image_size, image_size)),

    indices=perm[:train_data_size])

valid_dataset = BengaliAIDataset(

    train_images, train_labels, transform=Transform(affine=False, crop=True, size=(image_size, image_size)),

    indices=perm[train_data_size:train_data_size+valid_data_size])

print('train_dataset', len(train_dataset), 'valid_dataset', len(valid_dataset))

# --- Model ---

device = torch.device(device)

n_grapheme = 168

n_vowel = 11

n_consonant = 7

n_total = n_grapheme + n_vowel + n_consonant

print('n_total', n_total)

# Set pretrained='imagenet' to download imagenet pretrained model...

predictor = PretrainedCNN(in_channels=1, out_dim=n_total, model_name=model_name, pretrained=None)

print('predictor', type(predictor))



classifier = BengaliClassifier(predictor).to(device)
import json

from logging import getLogger

import numpy



def save_json(filepath, params):

    with open(filepath, 'w') as f:

        json.dump(params, f, indent=4)
import os

from logging import getLogger

from time import perf_counter



import pandas as pd

import torch

# from chainer_chemistry.utils import save_json



from ignite.engine.engine import Engine, Events

from ignite.metrics import Average





class DictOutputTransform:

    def __init__(self, key, index=0):

        self.key = key

        self.index = index



    def __call__(self, x):

        if self.index >= 0:

            x = x[self.index]

        return x[self.key]





def create_trainer(classifier, optimizer, device):

    classifier.to(device)



    def update_fn(engine, batch):

        classifier.train()

        optimizer.zero_grad()

        # batch = [elem.to(device) for elem in batch]

        x, y = [elem.to(device) for elem in batch]

        loss, metrics, pred_y = classifier(x, y)

        loss.backward()

        optimizer.step()

        return metrics, pred_y, y

    trainer = Engine(update_fn)



    for key in classifier.metrics_keys:

        Average(output_transform=DictOutputTransform(key)).attach(trainer, key)

    return trainer





def create_evaluator(classifier, device):

    classifier.to(device)



    def update_fn(engine, batch):

        classifier.eval()

        with torch.no_grad():

            # batch = [elem.to(device) for elem in batch]

            x, y = [elem.to(device) for elem in batch]

            _, metrics, pred_y = classifier(x, y)

            return metrics, pred_y, y

    evaluator = Engine(update_fn)



    for key in classifier.metrics_keys:

        Average(output_transform=DictOutputTransform(key)).attach(evaluator, key)

    return evaluator





class LogReport:

    def __init__(self, evaluator=None, dirpath=None, logger=None):

        self.evaluator = evaluator

        self.dirpath = str(dirpath) if dirpath is not None else None

        self.logger = logger or getLogger(__name__)



        self.reported_dict = {}  # To handle additional parameter to monitor

        self.history = []

        self.start_time = perf_counter()



    def report(self, key, value):

        self.reported_dict[key] = value



    def __call__(self, engine):

        elapsed_time = perf_counter() - self.start_time

        elem = {'epoch': engine.state.epoch,

                'iteration': engine.state.iteration}

        elem.update({f'train/{key}': value

                     for key, value in engine.state.metrics.items()})

        if self.evaluator is not None:

            elem.update({f'valid/{key}': value

                         for key, value in self.evaluator.state.metrics.items()})

        elem.update(self.reported_dict)

        elem['elapsed_time'] = elapsed_time

        self.history.append(elem)

        if self.dirpath:

            save_json(os.path.join(self.dirpath, 'log.json'), self.history)

            self.get_dataframe().to_csv(os.path.join(self.dirpath, 'log.csv'), index=False)



        # --- print ---

        msg = ''

        for key, value in elem.items():

            if key in ['iteration']:

                # skip printing some parameters...

                continue

            elif isinstance(value, int):

                msg += f'{key} {value: >6d} '

            else:

                msg += f'{key} {value: 8f} '

#         self.logger.warning(msg)

        print(msg)



        # --- Reset ---

        self.reported_dict = {}



    def get_dataframe(self):

        df = pd.DataFrame(self.history)

        return df





class SpeedCheckHandler:

    def __init__(self, iteration_interval=10, logger=None):

        self.iteration_interval = iteration_interval

        self.logger = logger or getLogger(__name__)

        self.prev_time = perf_counter()



    def __call__(self, engine: Engine):

        if engine.state.iteration % self.iteration_interval == 0:

            cur_time = perf_counter()

            spd = self.iteration_interval / (cur_time - self.prev_time)

            self.logger.warning(f'{spd} iter/sec')

            # reset

            self.prev_time = cur_time



    def attach(self, engine: Engine):

        engine.add_event_handler(Events.ITERATION_COMPLETED, self)





class ModelSnapshotHandler:

    def __init__(self, model, filepath='model_{count:06}.pt',

                 interval=1, logger=None):

        self.model = model

        self.filepath: str = str(filepath)

        self.interval = interval

        self.logger = logger or getLogger(__name__)

        self.count = 0



    def __call__(self, engine: Engine):

        self.count += 1

        if self.count % self.interval == 0:

            filepath = self.filepath.format(count=self.count)

            torch.save(self.model.state_dict(), filepath)

            # self.logger.warning(f'save model to {filepath}...')

import warnings



import torch



from ignite.metrics.metric import Metric





class EpochMetric(Metric):

    """Class for metrics that should be computed on the entire output history of a model.

    Model's output and targets are restricted to be of shape `(batch_size, n_classes)`. Output

    datatype should be `float32`. Target datatype should be `long`.



    .. warning::



        Current implementation stores all input data (output and target) in as tensors before computing a metric.

        This can potentially lead to a memory error if the input data is larger than available RAM.





    - `update` must receive output of the form `(y_pred, y)`.



    If target shape is `(batch_size, n_classes)` and `n_classes > 1` than it should be binary: e.g. `[[0, 1, 0, 1], ]`.



    Args:

        compute_fn (callable): a callable with the signature (`torch.tensor`, `torch.tensor`) takes as the input

            `predictions` and `targets` and returns a scalar.

        output_transform (callable, optional): a callable that is used to transform the

            :class:`~ignite.engine.Engine`'s `process_function`'s output into the

            form expected by the metric. This can be useful if, for example, you have a multi-output model and

            you want to compute the metric with respect to one of the outputs.



    """



    def __init__(self, compute_fn, output_transform=lambda x: x):



        if not callable(compute_fn):

            raise TypeError("Argument compute_fn should be callable.")



        super(EpochMetric, self).__init__(output_transform=output_transform)

        self.compute_fn = compute_fn



    def reset(self):

        self._predictions = torch.tensor([], dtype=torch.float32)

        self._targets = torch.tensor([], dtype=torch.long)



    def update(self, output):

        y_pred, y = output

        self._predictions = torch.cat([self._predictions, y_pred], dim=0)

        self._targets = torch.cat([self._targets, y], dim=0)



        # Check once the signature and execution of compute_fn

        if self._predictions.shape == y_pred.shape:

            try:

                self.compute_fn(self._predictions, self._targets)

            except Exception as e:

                warnings.warn("Probably, there can be a problem with `compute_fn`:\n {}.".format(e),

                              RuntimeWarning)



    def compute(self):

        return self.compute_fn(self._predictions, self._targets)

import numpy as np

import sklearn.metrics

import torch





def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):

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

import argparse

from distutils.util import strtobool

import os



import torch

from ignite.contrib.handlers import ProgressBar

from ignite.engine import Events

from numpy.random.mtrand import RandomState

from torch.utils.data.dataloader import DataLoader



# --- Training setting ---



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)



optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-10)



trainer = create_trainer(classifier, optimizer, device)

def output_transform(output):

    metric, pred_y, y = output

    return pred_y.cpu(), y.cpu()

EpochMetric(

    compute_fn=macro_recall,

    output_transform=output_transform

).attach(trainer, 'recall')



pbar = ProgressBar()

pbar.attach(trainer, metric_names='all')



evaluator = create_evaluator(classifier, device)

EpochMetric(

    compute_fn=macro_recall,

    output_transform=output_transform

).attach(evaluator, 'recall')



def run_evaluator(engine):

    evaluator.run(valid_loader)



def schedule_lr(engine):

    # metrics = evaluator.state.metrics

    metrics = engine.state.metrics

    avg_mae = metrics['loss']



    # --- update lr ---

    lr = scheduler.optimizer.param_groups[0]['lr']

    scheduler.step(avg_mae)

    log_report.report('lr', lr)



trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator)

trainer.add_event_handler(Events.EPOCH_COMPLETED, schedule_lr)

log_report = LogReport(evaluator, outdir)

trainer.add_event_handler(Events.EPOCH_COMPLETED, log_report)

trainer.add_event_handler(

    Events.EPOCH_COMPLETED,

    ModelSnapshotHandler(predictor, filepath=outdir / 'predictor.pt'))

trainer.run(train_loader, max_epochs=10)
train_history = log_report.get_dataframe()

train_history.to_csv(outdir / 'log.csv', index=False)



train_history