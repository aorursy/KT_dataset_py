%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from PIL import Image

import os

import torch

import copy

import torch.utils.data

import matplotlib.patches as patches

from torchvision import transforms

import collections

import random

from tqdm import tqdm_notebook as tqdm

print(os.listdir("../input"))
def rle2mask(rle, width, height):

    mask= np.zeros(width* height)

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]

    current_position = 0

    for index, start in enumerate(starts):

        current_position += start

        mask[current_position:current_position+lengths[index]] = 1

        current_position += lengths[index]

    return mask.reshape(width, height)
class SIIMDataset(torch.utils.data.Dataset):

    def __init__(self, df_path, img_dir):

        self.df = pd.read_csv(df_path, nrows=100)

        self.height = 1024

        self.width = 1024

        self.image_dir = img_dir

        self.image_info = collections.defaultdict(dict)

        self.df = self.df.drop_duplicates('ImageId', keep='last').reset_index(drop=True)



        counter = 0

        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):

            image_id = row['ImageId']

            image_path = os.path.join(self.image_dir, image_id)

            if os.path.exists(image_path + '.png') and row[" EncodedPixels"].strip() != "-1":

                self.image_info[counter]["image_id"] = image_id

                self.image_info[counter]["image_path"] = image_path

                self.image_info[counter]["annotations"] = row[" EncodedPixels"].strip()

                counter += 1



    def __getitem__(self, idx):

        img_path = self.image_info[idx]["image_path"]

        img = Image.open(img_path + '.png').convert("RGB")

        width, height = img.size

        img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        info = self.image_info[idx]



        mask = rle2mask(info['annotations'], width, height)

        mask = Image.fromarray(mask.T)

        mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)

        mask = np.expand_dims(mask, axis=0)



        pos = np.where(np.array(mask)[0, :, :])

        xmin = np.min(pos[1])

        xmax = np.max(pos[1])

        ymin = np.min(pos[0])

        ymax = np.max(pos[0])



        boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        labels = torch.ones((1,), dtype=torch.int64)

        masks = torch.as_tensor(mask, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((1,), dtype=torch.int64)



        target = {}

        target["boxes"] = boxes

        target["labels"] = labels

        target["masks"] = masks

        target["image_id"] = image_id

        target["area"] = area

        target["iscrowd"] = iscrowd

        return transforms.ToTensor()(img), target



    def __len__(self):

        return len(self.image_info)
dataset = SIIMDataset("../input/siim-dicom-images/train-rle.csv", "../input/siim-png-images/input/train_png/")
def plotter(img, target):

    fig,ax = plt.subplots(1)

    ax.imshow(transforms.ToPILImage()(img))

    ax.imshow(transforms.ToPILImage()(target["masks"][0].cpu() * 255), alpha=0.5)

    rect = patches.Rectangle((target["boxes"][0][0].item(),

                             target["boxes"][0][1].item()),

                             target["boxes"][0][2].item() - target["boxes"][0][0].item(),

                             target["boxes"][0][3].item() - target["boxes"][0][1].item(),

                             linewidth=1,edgecolor='r',facecolor='none')

    ax.add_patch(rect)

    plt.show()
img, target = dataset[0]

plotter(img, target)
def horizontal_flip(image, target):

    img = copy.deepcopy(image)

    tgt = copy.deepcopy(target)

    height, width = img.shape[-2:]

    img = img.flip(-1)

    bbox = tgt["boxes"]

    bbox[:, [0, 2]] = width - bbox[:, [2, 0]]

    tgt["boxes"] = bbox

    tgt["masks"] = tgt["masks"].flip(-1)

    return img, tgt
_img, _target = horizontal_flip(img, target)

plotter(_img, _target)
def vertical_flip(image, target):

    img = copy.deepcopy(image)

    tgt = copy.deepcopy(target)

    height, width = img.shape[-2:]

    img = img.flip(1)

    bbox = tgt["boxes"]

    bbox[:, [1, 3]] = height - bbox[:, [3, 1]]

    tgt["boxes"] = bbox

    tgt["masks"] = tgt["masks"].flip(1)

    return img, tgt
_img, _target = vertical_flip(img, target)

plotter(_img, _target)
def rotation(image, target, degrees):

    img = copy.deepcopy(image)

    tgt = copy.deepcopy(target)

    height, width = img.shape[-2:]

    img = transforms.ToTensor()(transforms.ToPILImage()(img).rotate(degrees))

    rotated_mask = transforms.ToTensor()(transforms.ToPILImage()(tgt["masks"]).rotate(degrees))

    

    pos = np.where(np.array(rotated_mask)[0, :, :])

    xmin = np.min(pos[1])

    xmax = np.max(pos[1])

    ymin = np.min(pos[0])

    ymax = np.max(pos[0])



    bbox = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

    

    tgt["boxes"] = bbox

    tgt["masks"] = rotated_mask

    return img, tgt
_img, _target = rotation(img, target, 30)

plotter(_img, _target)
_img, _target = rotation(img, target, 90)

plotter(_img, _target)
_img, _target = rotation(img, target, 120)

plotter(_img, _target)
_img, _target = rotation(img, target, 270)

plotter(_img, _target)