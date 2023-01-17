import numpy as np
import pandas as pd
import gc

import warnings
warnings.filterwarnings('ignore')

import os
import glob
import os.path as osp
from PIL import Image

import matplotlib.pyplot as plt
%matplotlib inline
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data as D
# Tutorail for torch version:
torch.__version__
path = '../input/train/'
class AirbusDS(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, root):
        """ Intialize the dataset
        """
        self.filenames = []
        self.root = root
        self.transform = transforms.ToTensor()
        filenames = glob.glob(osp.join(path, '*.jpg'))
        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)
        
    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        image = Image.open(self.filenames[index])
        return self.transform(image)

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
# Simple dataset. Only save path to image and load it and transform to tensor when call __getitem__.
airimg = AirbusDS(path)
# total images in set
print(airimg.len)
# Use the torch dataloader to iterate through the dataset
loader = D.DataLoader(airimg, batch_size=24, shuffle=False, num_workers=0)

# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some images
dataiter = iter(loader)
images = dataiter.next()

# show images
plt.figure(figsize=(16,8))
imshow(torchvision.utils.make_grid(images))
train_len = int(0.7*airimg.len)
valid_len = airimg.len - train_len
train, valid = D.random_split(airimg, lengths=[train_len, valid_len])
# check lens of subset
len(train), len(valid)
# https://github.com/albu/albumentations
from albumentations import (ToFloat, 
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, 
    Flip, OneOf, Compose
)
class AirbusDS(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, root, aug = False):
        """ Intialize the dataset
        """
        self.filenames = []
        self.root = root
        self.aug = aug
        if self.aug:
            self.transform = OneOf([
                                CLAHE(clip_limit=2),
                                IAASharpen(),
                                RandomRotate90(),
                                IAAEmboss(),
                                Transpose(),
                                RandomContrast(),
                                RandomBrightness(),
                            ], p=0.3)
        else:
            self.transform = transforms.ToTensor()
        filenames = glob.glob(osp.join(path, '*.jpg'))
        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)
        
    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        image = Image.open(self.filenames[index])
        if self.aug:
            data = {"image": np.array(image)}
            image = self.transform(**data)['image']
            images = np.transpose(image, (2, 0, 1))
            return images
        else:
            return self.transform(image)

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
airimg = AirbusDS(path, aug=True)
# Use the torch dataloader to iterate through the dataset
loader = D.DataLoader(airimg, batch_size=24, shuffle=False, num_workers=0)

# get some images
dataiter = iter(loader)
images = dataiter.next()

# show images
plt.figure(figsize=(16,8))
imshow(torchvision.utils.make_grid(images))
# based on https://www.kaggle.com/inversion/run-length-decoding-quick-start
masks = pd.read_csv('../input/train_ship_segmentations.csv')
masks.head()
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction
class AirbusDS(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, root, aug = False, mode='train'):
        """ Intialize the dataset
        """
        self.filenames = []
        self.root = root
        self.aug = aug
        self.mode = 'test'
        if mode == 'train':
            self.mode = 'train'
            self.masks = pd.read_csv('../input/train_ship_segmentations.csv').fillna(-1)
        if self.aug:
            self.transform = OneOf([
                                RandomRotate90(),
                                Transpose(),
                                Flip(),
                            ], p=0.3)
        else:
            self.transform = transforms.ToTensor()
        filenames = glob.glob(osp.join(path, '*.jpg'))
        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)
        
    # You must override __getitem__ and __len__
    def get_mask(self, ImageId):
        img_masks = self.masks.loc[self.masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

        # Take the individual ship masks and create a single mask array for all ships
        all_masks = np.zeros((768, 768))
        if img_masks == [-1]:
            return all_masks
        for mask in img_masks:
            all_masks += rle_decode(mask)
        return all_masks
    
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        image = Image.open(self.filenames[index])
        ImageId = self.filenames[index].split('/')[-1]
        if self.mode == 'train':
            mask = self.get_mask(ImageId)
        if self.aug:
            if self.mode == 'train':
                data = {"image": np.array(image), "mask": mask}
            else:
                data = {"image": np.array(image)}
            transformed = self.transform(**data)
            image = transformed['image']/255
            image = np.transpose(image, (2, 0, 1))
            if self.mode == 'train':
                return image, transformed['mask'][np.newaxis,:,:] 
            else:
                return image
        else:
            if self.mode == 'train':
                return self.transform(image), mask[np.newaxis,:,:] 
            return self.transform(image)

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
airimg = AirbusDS(path, aug=True, mode='train')
# Use the torch dataloader to iterate through the dataset
loader = D.DataLoader(airimg, batch_size=24, shuffle=False, num_workers=0)

# get some images
dataiter = iter(loader)
images, masks = dataiter.next()

# show images
plt.figure(figsize=(16,16))
plt.subplot(211)
imshow(torchvision.utils.make_grid(images))
plt.subplot(212)
imshow(torchvision.utils.make_grid(masks))
