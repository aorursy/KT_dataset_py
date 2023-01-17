# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

im = Image.open("../input/wildlife-images-kangaroo/Wildlife_Kangaroo/images/00003.jpg")



plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/wildlife-images-kangaroo/Wildlife_Kangaroo/images/00006.jpg")



plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/wildlife-images-kangaroo/Wildlife_Kangaroo/images/00010.jpg")



plt.imshow(im)

display(plt.show())
from PIL import Image

im = Image.open("../input/wildlife-images-kangaroo/Wildlife_Kangaroo/images/00013.jpg")



plt.imshow(im)

display(plt.show())
from fastai.vision import *
tfms = get_transforms(max_rotate=25)
len(tfms)
def get_ex(): return open_image('../input/wildlife-images-kangaroo/Wildlife_Kangaroo/images/00009.jpg')
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
# dihedral

fig, axs = plt.subplots(2,2,figsize=(12,8))

for k, ax in enumerate(axs.flatten()):

    dihedral(get_ex(), k).show(ax=ax, title=f'k={k}')

plt.tight_layout()