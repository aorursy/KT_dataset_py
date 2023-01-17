# !pip install pretrainedmodels  # <- need internet connection

# Ref: https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88 for installation



!pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null # no output

# !pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/
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
debug=False

submission=True

batch_size=256

device='cuda:0'

out='.'
datadir = Path('/kaggle/input/bengaliai-cv19')

featherdir = Path('/kaggle/input/bengaliaicv19feather')

outdir = Path('.')
# Read in the data CSV files

# train = pd.read_csv(datadir/'train.csv')

# test = pd.read_csv(datadir/'test.csv')

# sample_submission = pd.read_csv(datadir/'sample_submission.csv')

# class_map = pd.read_csv(datadir/'class_map.csv')
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

        # Opposite white and black: background will be white (1.0) and

        # for future Affine transformation

        x = (255 - x).astype(np.float32) / 255.

        if self.train:

            y = self.labels[i]

            return x, y

        else:

            return x

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





def crop_char_image(image, threshold=40./255.):

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

import numpy as np





def add_gaussian_noise(x, sigma):

    x += np.random.randn(*x.shape) * sigma

    x = np.clip(x, 0., 1.)

    return x





class Transform:

    def __init__(self, affine=True, crop=True, size=(64, 64),

                 normalize=True, train=True, threshold=40.,

                 sigma=-1.):

        self.affine = affine

        self.crop = crop

        self.size = size

        self.normalize = normalize

        self.train = train

        self.threshold = threshold / 255.

        self.sigma = sigma / 255.



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

                 pretrained=None):

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

def build_predictor(arch, out_dim, model_name=None):

    if arch == 'pretrained':

        predictor = PretrainedCNN(in_channels=1, out_dim=out_dim, model_name=model_name)

    else:

        raise ValueError("[ERROR] Unexpected value arch={}".format(arch))

    return predictor





def build_classifier(arch, load_model_path, n_total, model_name='', device='cuda:0'):

    if isinstance(device, str):

        device = torch.device(device)

    predictor = build_predictor(arch, out_dim=n_total, model_name=model_name)

    print('predictor', type(predictor))

    classifier = BengaliClassifier(predictor)

    if load_model_path:

        predictor.load_state_dict(torch.load(load_model_path))

    else:

        print("[WARNING] Unexpected value load_model_path={}"

              .format(load_model_path))

    classifier.to(device)

    return classifier

import gc

import numpy as np

import pandas as pd





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

def predict_core(test_images, image_size, threshold,

                 arch, n_total, model_name, load_model_path, batch_size=512, device='cuda:0', **kwargs):

    classifier = build_classifier(arch, load_model_path, n_total, model_name, device=device)

    test_dataset = BengaliAIDataset(

        test_images, None,

        transform=Transform(affine=False, crop=True, size=(image_size, image_size),

                            threshold=threshold, train=False))

    print('test_dataset', len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_pred_proba = classifier.predict_proba(test_loader)

    return test_pred_proba

# --- Model ---

device = torch.device(device)

n_grapheme = 168

n_vowel = 11

n_consonant = 7

n_total = n_grapheme + n_vowel + n_consonant

print('n_total', n_total)
from torch.utils.data.dataloader import DataLoader

from chainer_chemistry.utils import save_json, load_json





# --- Prediction ---

traindir = '/kaggle/input/bengaliaicv19-trainedmodels/'

data_type = 'test'

test_preds_list = []



for i in range(4):

    # --- prepare data ---

    indices = [i]

    test_images = prepare_image(

        datadir, featherdir, data_type=data_type, submission=submission, indices=indices)

    n_dataset = len(test_images)

    print(f'n_dataset={n_dataset}')

    # print(f'i={i}, n_dataset={n_dataset}')

    # test_data_size = 200 if debug else int(n_dataset * 0.9)



    model_preds_list = []

    for j in range(4):

        # --- Depends on train configuration ---

        train_args_dict = load_json(os.path.join(traindir, f'args_{j}.json'))

        train_args_dict.update({

            'load_model_path': os.path.join(traindir, f'predictor_{j}.pt'),

            'device': device,

            'batch_size': batch_size,

            'debug': debug,

        })

        print(f'j {j} updated train_args_dict {train_args_dict}')

        test_preds = predict_core(

                test_images=test_images, n_total=n_total,

                **train_args_dict)



        model_preds_list.append(test_preds)



    # --- ensemble ---

    proba0 = torch.mean(torch.stack([test_preds[0] for test_preds in model_preds_list], dim=0), dim=0)

    proba1 = torch.mean(torch.stack([test_preds[1] for test_preds in model_preds_list], dim=0), dim=0)

    proba2 = torch.mean(torch.stack([test_preds[2] for test_preds in model_preds_list], dim=0), dim=0)

    p0 = torch.argmax(proba0, dim=1).cpu().numpy()

    p1 = torch.argmax(proba1, dim=1).cpu().numpy()

    p2 = torch.argmax(proba2, dim=1).cpu().numpy()

    print('p0', p0.shape, 'p1', p1.shape, 'p2', p2.shape)



    test_preds_list.append([p0, p1, p2])

    if debug:

        break

    del test_images

    gc.collect()
p0 = np.concatenate([test_preds[0] for test_preds in test_preds_list], axis=0)

p1 = np.concatenate([test_preds[1] for test_preds in test_preds_list], axis=0)

p2 = np.concatenate([test_preds[2] for test_preds in test_preds_list], axis=0)

print('concat:', 'p0', p0.shape, 'p1', p1.shape, 'p2', p2.shape)



row_id = []

target = []

for i in tqdm(range(len(p0))):

    row_id += [f'Test_{i}_grapheme_root', f'Test_{i}_vowel_diacritic',

               f'Test_{i}_consonant_diacritic']

    target += [p0[i], p1[i], p2[i]]

submission_df = pd.DataFrame({'row_id': row_id, 'target': target})

submission_df.to_csv('submission.csv', index=False)

submission_df
train = pd.read_csv(datadir/'train.csv')
pred_df = pd.DataFrame({

    'grapheme_root': p0,

    'vowel_diacritic': p1,

    'consonant_diacritic': p2

})
fig, axes = plt.subplots(2, 3, figsize=(22, 6))

plt.title('Label Count')

sns.countplot(x="grapheme_root",data=train, ax=axes[0, 0])

sns.countplot(x="vowel_diacritic",data=train, ax=axes[0, 1])

sns.countplot(x="consonant_diacritic",data=train, ax=axes[0, 2])

sns.countplot(x="grapheme_root",data=pred_df, ax=axes[1, 0])

sns.countplot(x="vowel_diacritic",data=pred_df, ax=axes[1, 1])

sns.countplot(x="consonant_diacritic",data=pred_df, ax=axes[1, 2])

plt.tight_layout()

plt.show()
train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

sns.distplot(train_labels[:, 0], ax=axes[0], color='green', kde=False, label='train grapheme')

sns.distplot(train_labels[:, 1], ax=axes[1], color='green', kde=False, label='train vowel')

sns.distplot(train_labels[:, 2], ax=axes[2], color='green', kde=False, label='train consonant')

plt.tight_layout()
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

sns.distplot(p0, ax=axes[0], color='orange', kde=False, label='test grapheme')

sns.distplot(p1, ax=axes[1], color='orange', kde=False, label='test vowel')

sns.distplot(p2, ax=axes[2], color='orange', kde=False, label='test consonant')

plt.legend()

plt.tight_layout()
