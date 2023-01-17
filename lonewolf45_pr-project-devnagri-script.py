import os
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2

import torch

import torch.nn as nn

import torchvision

from torchvision import transforms

from sklearn.model_selection import StratifiedKFold

import albumentations as A

from sklearn.model_selection import train_test_split
KERNEL_TYPE = "2x2ConvNet"

MEAN = [0.485, 0.456, 0.406] 

STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32

HEIGHT = 32

WIDTH = 32

N_EPOCHS = 10

N_FOLDS = 5

N_WORKERS = 4

INIT_LR = 0.01

RANDOM_STATE = 47

DATASET_DIR = "../input/devanagari-character-set/"
df = pd.read_csv(DATASET_DIR + "data.csv")

classes = df.character.unique()



idx2class = {i:class_name for i, class_name in enumerate(classes)}

class2idx = {class_name:i for i, class_name in enumerate(classes)}



df["character_id"] = df.character.map(class2idx)



skf = StratifiedKFold(N_FOLDS, shuffle = True, random_state = RANDOM_STATE)

for i_fold, (train_idx, val_idx) in enumerate(skf.split(df, df.character)):

    df.loc[val_idx, "fold"] = i_fold

df.fold = df.fold.astype(np.int)
class Albumentations():

    def __init__(self, augmentations):

        self.augmentations  = A.Compose(augmentations)

    def __call__(self, image):

        return self.augmentations(image = image)["image"]

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, folds , mode, transform = None, transform_orig = None):

        df = df[df.fold.isin(folds)].reset_index(drop = True)

        self.images= df.drop(['character','fold', "character_id"], axis = 1).values

        self.labels = df['character_id'].values

        self.mode = mode

        self.transforms = transform

        self.transforms_orig = transform_orig

        

    def __len__(self):

        return self.images.shape[0]

    

    def __mode__(self):

        return self.mode

    

    def __getitem__(self, index):

        image = self.images[index]

        label = self.labels[index]

        image = image.reshape((HEIGHT, WIDTH))

        image_orig = image.astype(np.float32).copy()

        if self.transforms:

            image = self.transforms(image)

        if self.transforms_orig:

            image_orig = self.transforms_orig(image)

        

        return torch.tensor(image), torch.tensor(image_orig), torch.tensor(label)
preprocess=[

    

]



augmentations = [

    A.PadIfNeeded(min_height=HEIGHT, min_width=WIDTH, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], always_apply=True),

    A.OneOf([

        A.ShiftScaleRotate(rotate_limit=90, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0], mask_value=[255, 255, 255], always_apply=True),

    ], 0.5)

]

transforms_train = transforms.Compose([

    np.uint8,

    Albumentations(preprocess + augmentations),

    transforms.ToTensor(),

#     transforms.Normalize(mean = MEAN, std = STD)

])



transforms_val = transforms.Compose([

    np.uint8,

    Albumentations(preprocess),

    transforms.ToTensor(),

#     transforms.Normalize(mean = MEAN, std = STD)

    

])

transforms_orig = transforms.Compose([

    np.uint8,

    Albumentations(preprocess),

    transforms.ToTensor(),   

])

df_show = df.sample(n = 100)



dataset_show = Dataset( df_show, [0,1,2,3],"train",transforms_train, transforms_orig )



from pylab import rcParams

rcParams["figure.figsize"] = 20,10

for i in range(2):

    f, axarr = plt.subplots(1,5)

    for p in range(5):

        idx = np.random.randint(0, len(dataset_show))

        img, img_orig, label = dataset_show[idx]

        axarr[p].imshow(img.transpose(0,1).transpose(1,2).squeeze())

        axarr[p].set_title("index: "+ str(idx)+"label: " + str(label.item()))