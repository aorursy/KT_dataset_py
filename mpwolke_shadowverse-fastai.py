#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
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
from fastai.vision import *
tfms = get_transforms(max_rotate=25)
len(tfms)
def get_ex(): return open_image('../input/shadowverse-datset/img_amulet/101032010.jpeg')
def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]
plots_f(2, 4, 12, 6, size=224)
tfms = zoom_crop(scale=(0.75,2), do_rand=True)
# random zoom and crop

plots_f(2, 4, 12, 6, size=224)
# random resize and crop

tfms = [rand_resize_crop(224)]

plots_f(2, 4, 12, 6, size=224)
# passing a probability to a function

tfm = [rotate(degrees=30, p=0.5)]

fig, axs = plt.subplots(1,5,figsize=(12,4))

for ax in axs:

    img = get_ex().apply_tfms(tfm)

    title = 'Done' if tfm[0].do_run else 'Not done'

    img.show(ax=ax, title=title)
tfm = [rotate(degrees=(-30,30))]

fig, axs = plt.subplots(1,5,figsize=(12,4))

for ax in axs:

    img = get_ex().apply_tfms(tfm)

    title = f"deg={tfm[0].resolved['degrees']:.1f}"

    img.show(ax=ax, title=title)
# brightness

fig, axs = plt.subplots(1,5,figsize=(14,8))

for change, ax in zip(np.linspace(0.1,0.9,5), axs):

    brightness(get_ex(), change).show(ax=ax, title=f'change={change:.1f}')
# contrast

fig, axs = plt.subplots(1,5,figsize=(12,4))

for scale, ax in zip(np.exp(np.linspace(log(0.5),log(2),5)), axs):

    contrast(get_ex(), scale).show(ax=ax, title=f'scale={scale:.2f}')
# dihedral

fig, axs = plt.subplots(2,2,figsize=(12,8))

for k, ax in enumerate(axs.flatten()):

    dihedral(get_ex(), k).show(ax=ax, title=f'k={k}')

plt.tight_layout()
fig, axs = plt.subplots(1,2,figsize=(10,8))

get_ex().show(ax=axs[0], title=f'no flip')

flip_lr(get_ex()).show(ax=axs[1], title=f'flip')
# jitter

fig, axs = plt.subplots(1,5,figsize=(20,8))

for magnitude, ax in zip(np.linspace(-0.05,0.05,5), axs):

    tfm = jitter(magnitude=magnitude)

    get_ex().jitter(magnitude).show(ax=ax, title=f'magnitude={magnitude:.2f}')
# squish

fig, axs = plt.subplots(1,5,figsize=(12,4))

for scale, ax in zip(np.linspace(0.66,1.33,5), axs):

    get_ex().squish(scale=scale).show(ax=ax, title=f'scale={scale:.2f}')
# tilt

fig, axs = plt.subplots(2,4,figsize=(12,8))

for i in range(4):

    get_ex().tilt(i, 0.4).show(ax=axs[0,i], title=f'direction={i}, fwd')

    get_ex().tilt(i, -0.4).show(ax=axs[1,i], title=f'direction={i}, bwd')
# symm warp

tfm = symmetric_warp(magnitude=(-0.2,0.2))

_, axs = plt.subplots(2,4,figsize=(12,6))

for ax in axs.flatten():

    img = get_ex().apply_tfms(tfm, padding_mode='zeros')

    img.show(ax=ax)