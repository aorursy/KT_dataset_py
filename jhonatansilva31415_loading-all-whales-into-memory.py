from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import os
import copy

from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from torch.optim import lr_scheduler
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from IPython.display import clear_output

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
image_size = 128
humpback_whales_path = '/kaggle/input' 
train_path = os.path.join(humpback_whales_path,'train.csv')
humpback_whales_train_path = os.path.join(humpback_whales_path,'train')
class WhalesDS(Dataset):
    """ Humpback Whale Identification Challenge dataset. """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.whales_frame = self.encode()
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.whales_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.whales_frame.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.whales_frame.iloc[idx,1]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def encode(self):
        """ Encoding """
        df = pd.read_csv(train_path)
        unique_classes = pd.unique(df['Id'])
        encoding = dict(enumerate(unique_classes))
        encoding = {value: key for key, value in encoding.items()}
        df = df.replace(encoding)
        return df 
whales_ds = WhalesDS(csv_file=train_path,
                     root_dir=humpback_whales_train_path)

fig = plt.figure()

for i in range(len(whales_ds)):
    sample = whales_ds[i]
    print(i, sample['image'].shape, sample['label'])

    ax = plt.subplot(1, 4, i + 1)
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample['image'])

    if i == 3:
        plt.show()
        break
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        """ The original code didn't expect gray scale images """
        gray_scale_image = torch.zeros([image_size,image_size]).shape == image.shape
        if gray_scale_image:
            image = np.stack((image,)*3, axis=-1)
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.tensor(label)}
scale = Rescale(int(image_size*1.25))
crop = RandomCrop(image_size)
composed = transforms.Compose([Rescale(int(image_size*1.25)),
                               RandomCrop(image_size)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = whales_ds[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    ax.set_title(type(tsfrm).__name__)

    plt.imshow(transformed_sample['image'])
plt.show()
transformed_dataset = WhalesDS(csv_file=train_path,
                                           root_dir=humpback_whales_train_path,
                                           transform=transforms.Compose([
                                               Rescale(int(image_size*1.25)),
                                               RandomCrop(image_size),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['label'])

    if i == 3:
        break
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
# Helper function to show a batch
def show_whale_batch(sample_batched):
    """Show whales for a batch of samples."""
    images_batch, labels_batch = \
            sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['label'])
    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_whale_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break