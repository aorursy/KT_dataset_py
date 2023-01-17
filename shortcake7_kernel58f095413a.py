from shutil import copyfile



copyfile(src="../input/import-file/inceptionresnetv2.py", dst="../working/pretrainedmodels.py")
import glob

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.

import argparse

from itertools import islice

import json

from pathlib import Path

import shutil

import warnings

from typing import Dict

import os

import sys

from collections import OrderedDict

import math

import random

from typing import Callable, List

from datetime import datetime

import json

import glob

from multiprocessing.pool import ThreadPool

import gc



import torch

from torch import nn, cuda

from torch.nn import functional as F

import torch.utils.model_zoo as model_zoo

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from torchvision.transforms import (

            ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,

                RandomHorizontalFlip)



import tqdm

from PIL import Image

import cv2

cv2.setNumThreads(0)

from pretrainedmodels import *



def seed_everything(seed=1234):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

                               

seed_everything(7)



DATA_ROOT = Path('/kaggle/input/imet-2019-fgvc6')

N_CLASSES = 1103



test_transform = Compose([

    RandomCrop(320, pad_if_needed=True),

    RandomHorizontalFlip(),

    ])





tensor_transform = Compose([

    ToTensor(),

    Normalize(mean=[0.5949, 0.5611, 0.5185], std=[0.2900, 0.2844, 0.2811]),

    ])



class TTADataset:

    def __init__(self, root: Path, df: pd.DataFrame,

                 image_transform: Callable, tta: int):

        self._root = root

        self._df = df

        self._image_transform = image_transform

        self._tta = tta



    def __len__(self):

        return len(self._df) * self._tta



    def __getitem__(self, idx):

        item = self._df.iloc[idx % len(self._df)]

        image = load_transform_image(item, self._root, self._image_transform)

        return image, item.id





def load_transform_image(

        item, root: Path, image_transform: Callable, debug: bool = False):

    image = load_image(item, root)

    image = image_transform(image)

    if debug:

        image.save('_debug.png')

    return tensor_transform(image)





def train_load_transform_image(

        item, root: Path, image_transform: Callable, debug: bool = False):

    image = load_image(item, root)

    image = image_transform(image)

    if debug:

        image.save('_debug.png')

    return train_tensor_transform(image)





def load_image(item, root: Path) -> Image.Image:

    image = cv2.imread(str(root / f'{item.id}.png'))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(image)





def get_ids(root: Path) -> List[str]:

    return sorted({p.name.split('_')[0] for p in root.glob('*.png')})



def load_model(model: nn.Module, path: Path) -> Dict:

    state = torch.load(str(path))

    model.load_state_dict(state['model'])

    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))

    return state



def mean_df(df: pd.DataFrame) -> pd.DataFrame:

    return df.groupby(level=0).mean()



def get_classes(item):

    return ' '.join(cls for cls, is_present in item.items() if is_present)



def binarize_prediction(probabilities, threshold: float, argsorted=None,

                        min_labels=1, max_labels=10):

    """ Return matrix of 0/1 predictions, same shape as probabilities.

    """

    assert probabilities.shape[1] == N_CLASSES

    if argsorted is None:

        argsorted = probabilities.argsort(axis=1)

    max_mask = _make_mask(argsorted, max_labels)

    min_mask = _make_mask(argsorted, min_labels)

    

    prob_mask = []

    for prob in probabilities:

        prob_mask.append(prob > prob.max()/7)

        

    prob_mask = np.array(prob_mask, dtype=np.int)

    

    return (max_mask & prob_mask) | min_mask





def _make_mask(argsorted, top_n: int):

    mask = np.zeros_like(argsorted, dtype=np.uint8)

    col_indices = argsorted[:, -top_n:].reshape(-1)

    row_indices = [i // top_n for i in range(len(col_indices))]

    mask[row_indices, col_indices] = 1

    return mask



args = {

    'batch_size':16,

    'tta':2,

    'use_cuda':1,

    'workers':1,

    'threshold':0.1,

    'max_labels':10,

    'output':'/kaggle/working/submission.csv',

}



def create_model(model):

    feature_dim = model.last_linear.in_features

    class AvgPool(nn.Module):

        def forward(self, x):

            # print (x.size())

            return F.avg_pool2d(x, x.shape[2:])

    model.avg_pool = AvgPool()

    model.avgpool = AvgPool()

    model.last_linear = nn.Linear(feature_dim, N_CLASSES)

    model = torch.nn.DataParallel(model)

    model = model.cuda()

    return model

    

def test(model, loader, model_path, multi=False, half=False):

    load_model(model, model_path / 'best-model.pt')

    df = predict(model, loader, use_cuda=args['use_cuda'], half=half)

    return df

    

def predict(model, loader, use_cuda: bool, half=False):

    model.eval()

    all_outputs, all_ids = [], []

    with torch.no_grad():

        for inputs, ids in loader:

            inputs = inputs.cuda()

            outputs = torch.sigmoid(model(inputs))

            # outputs = model(inputs)

            all_outputs.append(outputs.detach().cpu().numpy())

            all_ids.extend(ids)

    df = pd.DataFrame(

        data=np.concatenate(all_outputs),

        index=all_ids,

        columns=map(str, range(N_CLASSES)))

    df = mean_df(df)

    return df



import string

def randomString2(stringLength=8):

    """Generate a random string of fixed length """

    letters= string.ascii_lowercase

    return ''.join(random.sample(letters,stringLength))





test_root = DATA_ROOT /'test'

test_df = pd.read_csv(DATA_ROOT / 'sample_submission.csv')

# df = pd.concat([df]*5, ignore_index=True)

# df['new_id'] = [randomString2() for i in range(len(df))]

loader = DataLoader(

        dataset=TTADataset(test_root, test_df, test_transform, tta=args['tta']),

        shuffle=False,

        batch_size=args['batch_size'],

        num_workers=args['workers'],

    )





import gc



dfs = []

model = inceptionresnetv2(pretrained=False)

model = create_model(model)

#for i in range(5,10):

for i in range(1):

    df = test(model, loader, Path(f'/kaggle/input/import-file/'), multi=True)

    gc.collect()

    dfs.append(df)

df = pd.concat(dfs)

df = mean_df(df)

out_path = '05_30_inres2.h5'

#df.to_hdf(out_path, 'prob', index_label='id')

print(f'Saved predictions to {out_path}')

pred = df.values



df[:] = binarize_prediction(pred, threshold=args['threshold'], max_labels=args['max_labels'])

df = df.apply(get_classes, axis=1)

df.name = 'attribute_ids'

df.to_csv(args['output'], header=True)

print(f'Saved submission.csv to' + args['output'])