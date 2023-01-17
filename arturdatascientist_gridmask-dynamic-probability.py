import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import gc



%matplotlib inline 



rng = np.random.default_rng()
TRAIN_PARQS = ['train_image_data_0.parquet']
DATA_FOLDER = '../input/bengaliai-cv19'



train_df = pd.read_csv(os.path.join(DATA_FOLDER, 'train.csv'))

train_df.head()
from torch.utils.data import Dataset

from torch.utils.data import DataLoader

import torch

import cv2
IMG_H = 137

IMG_W = 236

IMG_SIZE = 128

BATCH_SIZE = 64

DEBUG = True

RAND_SEED = 605

LABEL_CLS = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
np.random.seed(RAND_SEED)

torch.manual_seed(RAND_SEED)
def prepare_data(img_path, 

               label_df=None, 

               debug=DEBUG, 

               dbg_max_count=300,

               label_cls=LABEL_CLS):

    img_path = [img_path] if isinstance(img_path, str) else img_path

    parq_list = []



    for i_path in img_path:

      parq = pd.read_parquet(os.path.join(DATA_FOLDER, i_path))

      parq_list.append(parq.iloc[:, 1:].values.astype(np.uint8).reshape(-1, IMG_H, IMG_W))



      if debug: break



    images = np.concatenate(parq_list, axis=0)

    if debug:

      images = images[:dbg_max_count]

    

    labels = label_df if label_df is None else label_df[label_cls].values[:images.shape[0]]



    return images, labels

class SmartCrop:

  """Crop the image by light pixels edges"""



  def __init__(self, size, padding=15, x_marg=10, y_marg=10, mask_threshold=80, noise_threshold=30):

    self.size = size

    self.padding = padding

    self.x_marg = x_marg

    self.y_marg = y_marg

    self.noise_threshold = noise_threshold

    self.mask_threshold = mask_threshold



  def __call__(self, image):

    img_mask = (image > self.mask_threshold).astype(np.int)

    rows = np.any(img_mask, axis=1)

    cols = np.any(img_mask, axis=0)

    ymin, ymax = np.where(rows)[0][[0, -1]]

    xmin, xmax = np.where(cols)[0][[0, -1]]



    #cropping may cut too much, so we need to add it back

    xmin = xmin - self.x_marg if (xmin > self.x_marg) else 0

    ymin = ymin - self.y_marg if (ymin > self.y_marg) else 0

    xmax = xmax + self.x_marg if (xmax < IMG_W - self.x_marg) else IMG_W

    ymax = ymax + self.y_marg if (ymax < IMG_H - self.y_marg) else IMG_H

    image = image[ymin:ymax, xmin:xmax]

    image[image < self.noise_threshold] = 0 #remove low intensity pixels as noise



    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + self.padding



    #make sure that the aspect ratio is kept in rescaling

    image = np.pad(image, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(image, (self.size, self.size))



class InverseMaxNorm8bit:

  """Inverse np.uint8 pixels and scale with max"""

  def __call__(self, x):

    x_inv = 255 - x

    return (x_inv*(255/x_inv.max())).astype(np.uint8)



class Inverse8bit:

  """Inverse np.uint8 pixels"""

  def __call__(self, x):

    return (255 - x).astype(np.uint8)
class BengaliData(Dataset):

  """

    Bengali graphics data

  """



  def __init__(self, images, labels=None, transform=None):

    

    assert isinstance(images, np.ndarray)

    assert (images.shape[1], images.shape[2]) == (IMG_H, IMG_W)



    if labels is not None:

      assert isinstance(labels, np.ndarray)

      assert images.shape[0] == labels.shape[0]



    self.images = images

    self.labels = labels

    self.transform = transform

    self.inverse = InverseMaxNorm8bit()

    self.smart_crop = SmartCrop(IMG_SIZE)



  def __len__(self):

    return len(self.images)



  def __getitem__(self, idx):

        x = self.inverse(self.images[idx])

        x = self.smart_crop(x)



        if self.transform is not None:

            x = self.transform(image=x)['image']        



        x = torch.tensor(x).unsqueeze(0)



        if self.labels is not None:

            y = self.labels[idx]

            return x, torch.tensor(y)

        else:

            return x


img_avr = 0.07247582 #0.100065514 #0.06922848809290576

img_std = 0.20951288996105613 #0.2403564531227429 #0.20515700083327537
import albumentations as A

from albumentations import Normalize 

from albumentations.augmentations import functional as F

from albumentations.core import transforms_interface as TI
class GridMask(TI.ImageOnlyTransform):

    """GridMask augmentation for image classification and object detection.

    

    Author: @artur.k.space

    

    Args:

        ratio (int or (int, int)): ratio which define "l" size of grid units

        num_grid (int): number of grid in a row or column.

        fill_value (int, float, lisf of int, list of float): value for dropped pixels.

        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int

            an angle is picked from (-rotate, rotate). Default: (-90, 90)

        mode (int):

            0 - cropout a quarter of the square of each grid (left top)

            1 - reserve a quarter of the square of each grid (left top)

            2 - cropout 2 quarter of the square of each grid (left top & right bottom)



    Targets:

        image



    Image types:

        uint8, float32



    Reference:

    |  https://arxiv.org/abs/2001.04086

    |  https://github.com/akuxcw/GridMask

    |  https://www.kaggle.com/haqishen/gridmask

    """



    def __init__(self, ratio=(0.4, 0.7), num_grid=3, fill_value=0, rotate=90, mode=0, always_apply=False, p=0.5):

        super(GridMask, self).__init__(always_apply, p)



        if isinstance(num_grid, int):

            num_grid = (num_grid, num_grid)

        if isinstance(rotate, int):

            rotate = (-rotate, rotate)



        self.num_grid = num_grid

        self.fill_value = fill_value

        self.rotate = rotate

        self.mode = mode

        self.masks = None

        self.ratio = 0.5 if mode == 2 else ratio

        self.hh = None # diagonal

        self.height, self.width = None, None



    def init_masks(self, height, width):

        self.masks = []

        self.height, self.width = height, width

        self.hh = int(np.ceil(np.sqrt(height**2 + width**2)))



        for n, n_grid in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):

            self.masks.append(self.make_grid(n_grid))



    def make_grid(self, n_grid):

        assert self.hh is not None



        d_h = self.height / n_grid

        d_w = self.width / n_grid

                

        mask = np.ones((self.hh, self.hh), np.float32)

        r = np.random.uniform(self.ratio[0], self.ratio[1]) if isinstance(self.ratio, tuple) else self.ratio



        l_h = int(np.ceil(d_h*r))

        l_w = int(np.ceil(d_w*r))

        

        for i in range(-1, self.hh//int(d_h)+1):

            s = int(d_h*i + d_h)

            t = s+l_h

            s = max(min(s, self.hh), 0)

            t = max(min(t, self.hh), 0)



            if self.mode == 2:

                mask[s:t,:] = 1 - mask[s:t] # invert

            else:

                mask[s:t,:] = self.fill_value



        for i in range(-1, self.hh//int(d_w)+1):

            s = int(d_w*i + d_w)

            t = s+l_w

            s = max(min(s, self.hh), 0)

            t = max(min(t, self.hh), 0)



            if self.mode == 2:

                mask[:,s:t] = 1 - mask[:,s:t] # invert

            else:

                mask[:,s:t] = self.fill_value



        if self.mode == 1:

            mask = 1 - mask



        return mask 



    def apply(self, image, **params):

        h, w = image.shape[:2]



        if self.masks is None: self.init_masks(h, w)



        mask = rng.choice(self.masks)

        rand_h = np.random.randint(self.hh-h)

        rand_w = np.random.randint(self.hh-w)

        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0



        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask

        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask

        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)

        return image



    def get_transform_init_args_names(self):

        return ("ratio", "num_grid", "fill_value", "rotate", "mode")
x_train, y_train = prepare_data(TRAIN_PARQS, train_df, debug=DEBUG)
gm1 = GridMask(num_grid=(3, 8), ratio=(0.1, 0.2), mode=0, p=1, fill_value=0)

gm2 = GridMask(num_grid=(5, 7), mode=2, p=1)

gm3 = GridMask(num_grid=(3, 5), ratio=(0.5, 0.8), mode=1, p=1, fill_value=0)



transform = A.Compose([

   

    A.OneOf([gm1, gm2, gm3], p=1),



    Normalize(img_avr, img_std),

])
train_ds = BengaliData(x_train, y_train, transform=transform)
x_0, y_0 = train_ds[0]

print("Train DS len:", len(train_ds), type(x_0), 'device:', x_0.device,  x_0.dtype, x_0.shape, y_0)

nrow, ncol = 1, 5



fig, axes = plt.subplots(nrow, ncol, figsize=(18, 8))

axes = axes.flatten()

for i, ax in enumerate(axes):

    image, label = train_ds[0]

    ax.imshow(image[0], cmap='Greys')

    ax.set_title(f'label: {label}')

plt.tight_layout()

plt.show()
nrow, ncol = 3, 6



fig, axes = plt.subplots(nrow, ncol, figsize=(18, 8))

axes = axes.flatten()

for i, ax in enumerate(axes):

    image, label = train_ds[i]

    ax.imshow(image[0], cmap='Greys')

    ax.set_title(f'label: {label}')

plt.tight_layout()

plt.show()
fig, axes = plt.subplots(1, len(gm1.masks), figsize=(18, 8))

axes = axes.flatten()



for i, mask in enumerate(gm1.masks):

    axes[i].imshow(mask, cmap='Greys')



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(1, len(gm2.masks), figsize=(18, 8))

axes = axes.flatten()



for i, mask in enumerate(gm2.masks):

    axes[i].imshow(mask, cmap='Greys')



plt.tight_layout()

plt.show()
fig, axes = plt.subplots(1, len(gm3.masks), figsize=(18, 8))

axes = axes.flatten()



for i, mask in enumerate(gm3.masks):

    axes[i].imshow(mask, cmap='Greys')



plt.tight_layout()

plt.show()
class DynamicProb(TI.BasicTransform):

    """DynamicProb improvement

    

    Author: @artur.k.space

    

    Args:

        transform (BasicTransform): Albumentations transformer instance

        final_cnt (int): final instance calls count of the last step

        p_steps (int): progression steps number

        p (tuple(float)): min prob, max prob

    """

    def __init__(self, transform: TI.BasicTransform, final_cnt: int, p_steps: int = 5, p: tuple = (0, 0.8), always_apply = False):

        super().__init__(always_apply, p=1)



        transform.always_apply = True

        self.transform = transform



        self._cnt = 0

        self._stepid = 0

        self._prob = p[0]

        self.p_steps = p_steps

        self.c_range = np.linspace(0, final_cnt, num=p_steps, dtype=np.int32)

        self.p_range = np.linspace(p[0], p[1], num=p_steps, dtype=np.float32)



    def _update_p(self):

        self._stepid += 1

        self._prob = self.p_range[self._stepid]



    def __call__(self, force_apply=False, **kwargs):

        result = kwargs 

        

        if self._prob > np.random.rand():

            result = self.transform(**kwargs)



        self._cnt += 1



        if self.p_steps - 1 > self._stepid and self._cnt >= self.c_range[self._stepid+1]:

            self._update_p()



        return result
steps_num = 4
gm = DynamicProb(GridMask(num_grid=(5, 7), mode=2, p=1), final_cnt=int(BATCH_SIZE*(steps_num-1)), p_steps=steps_num, p=(0, 1))

transform = A.Compose([

    gm,

    A.Normalize(img_avr, img_std),

])
train_ds = BengaliData(x_train, y_train, transform=transform)

train_ds = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

train_iter = iter(train_ds)
nrow, ncol = 3, 6



for i in range(steps_num):

    print("Batch #%s GM prob %s" % (i, gm._prob))



    batch, labels = next(train_iter)

    

    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 5))

    axes = axes.flatten()

    for i, ax in enumerate(axes):

        image, label = batch[i], labels[i]

        ax.imshow(image[0], cmap='Greys')

        ax.set_title(f'label: {label}')

    plt.tight_layout()

    plt.show()
# For using with other transformers you can do smthng like this:



EPOCHS = 50

gm_final_cnt = int((EPOCHS-1)*len(x_train)*0.33)



gm1 = DynamicProb(GridMask(num_grid=(3, 8), ratio=(0.1, 0.2), mode=0, p=1, fill_value=0), final_cnt=gm_final_cnt, p_steps=steps_num, p=(0, 0.9))

gm2 = DynamicProb(GridMask(num_grid=(5, 7), mode=2, p=1),                                 final_cnt=gm_final_cnt, p_steps=steps_num, p=(0, 0.8))

gm3 = DynamicProb(GridMask(num_grid=(3, 5), ratio=(0.5, 0.8), mode=1, p=1, fill_value=0), final_cnt=gm_final_cnt, p_steps=steps_num, p=(0, 0.7))

A.OneOf([gm1, gm2, gm3], p=1)