# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import fastai

from fastai.tabular.all import *

from fastai.text.all import *

from fastai.vision.all import *

from fastai.medical.imaging import *

from fastai import *



import time

from datetime import datetime



print(f'Notebook last run on {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')

print('Using fastai version ',fastai.__version__)

print('And torch version ',torch.__version__)
from PIL import Image



img = Image.open("../input/maleandfemale/data1/validation/male/7733.jpg")

img

def _add1(x): return x+1

dumb_tfm = RandTransform(enc=_add1, p=0.5)

start,d1,d2 = 2,False,False

for _ in range(40):

    t = dumb_tfm(start, split_idx=0)

    if dumb_tfm.do: test_eq(t, start+1); d1=True

    else:           test_eq(t, start)  ; d2=True

assert d1 and d2

dumb_tfm
_,axs = subplots(1,2)

show_image(img, ctx=axs[0], title='original')

show_image(img.flip_lr(), ctx=axs[1], title='flipped');
tflip = FlipItem(p=1.)

test_eq(tflip(bbox,split_idx=0), tensor([[1.,0., 0.,1]]) -1)
_,axs = plt.subplots(1,3,figsize=(12,4))

for ax,sz in zip(axs.flatten(), [300, 500, 700]):

    show_image(img.crop_pad(sz), ctx=ax, title=f'Size {sz}');
_,axs = plt.subplots(1,3,figsize=(12,4))

for ax,mode in zip(axs.flatten(), [PadMode.Zeros, PadMode.Border, PadMode.Reflection]):

    show_image(img.crop_pad((600,700), pad_mode=mode), ctx=ax, title=mode);
_,axs = plt.subplots(1,3,figsize=(12,4))

f = RandomCrop(200)

for ax in axs: show_image(f(img), ctx=ax);
_,axs = plt.subplots(1,3,figsize=(12,4))

for ax in axs: show_image(f(img, split_idx=1), ctx=ax);
test_eq(ResizeMethod.Squish, 'squish')
Resize(224)
_,axs = plt.subplots(1,3,figsize=(12,4))

for ax,method in zip(axs.flatten(), [ResizeMethod.Squish, ResizeMethod.Pad, ResizeMethod.Crop]):

    rsz = Resize(256, method=method)

    show_image(rsz(img, split_idx=0), ctx=ax, title=method);
_,axs = plt.subplots(1,3,figsize=(12,4))

for ax,method in zip(axs.flatten(), [ResizeMethod.Squish, ResizeMethod.Pad, ResizeMethod.Crop]):

    rsz = Resize(256, method=method)

    show_image(rsz(img, split_idx=1), ctx=ax, title=method);
crop = RandomResizedCrop(256)

_,axs = plt.subplots(3,3,figsize=(9,9))

for ax in axs.flatten():

    cropped = crop(img)

    show_image(cropped, ctx=ax);
_,axs = subplots(1,3)

for ax in axs.flatten(): show_image(crop(img, split_idx=1), ctx=ax);
test_eq(cropped.shape, [256,256])
RatioResize(256)(img)
test_eq(RatioResize(256)(img).size[0], 256)

test_eq(RatioResize(256)(img.dihedral(3)).size[1], 256)
timg = TensorImage(array(img)).permute(2,0,1).float()/255.

def _batch_ex(bs): return TensorImage(timg[None].expand(bs, *timg.shape).clone())
t = _batch_ex(8)

rrc = RandomResizedCropGPU(224, p=1.)

y = rrc(t)

_,axs = plt.subplots(2,4, figsize=(12,6))

for ax in axs.flatten():

    show_image(y[i], ctx=ax)
x = flip_mat(torch.randn(100,4,3))

test_eq(set(x[:,0,0].numpy()), {-1,1}) #might fail with probability 2*2**(-100) (picked only 1s or -1s)