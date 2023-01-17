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
def get_ex(): return open_image('../input/indonesian-batik-motifs/batik-bali/1.jpg')
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