import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

import cv2

import os
train_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1]).astype('int8')

train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

train_df.drop(['ImageId_ClassId'], axis=1, inplace=True)

train_df.fillna(0, inplace=True)

train_df.head()
#simplified version of rle2mask() that returns total number of pixels

def pixel_count(rle):

    

    if np.logical_or(rle==0, rle==''):

        return(np.nan)

    array = np.asarray([int(x) for x in rle.split()])

    lengths = array[1::2]

    

    return sum(lengths)
DEFECT_TYPES = [1, 2, 3, 4]



for x in DEFECT_TYPES:

    train_df.loc[train_df.ClassId==x, 'pixel_count'] = train_df.loc[train_df.ClassId==x, 'EncodedPixels'].apply(pixel_count) 
train_df.head()
for x in DEFECT_TYPES:

    df = train_df.loc[train_df.ClassId==x, :]

    ax = df['pixel_count'].plot(kind='hist', bins=100, figsize=(15, 6))

    df['pixel_count'].plot(kind='kde', ax=ax, secondary_y=True)

    plt.xlim(left=0)

    plt.axvline(x=3500, linewidth=2, color='r')

    plt.title(f'Pixel Distribution for Defect Type {x}')

    print(f'=====\nDEFECT TYPE {x}\n=====')

    print(df['pixel_count'].describe())

    plt.show()
THRESHOLDS = []



for x in DEFECT_TYPES:

    pixels = train_df.loc[train_df['ClassId']==x, 'pixel_count']

    threshold = np.nanquantile(pixels, 0.1)

    THRESHOLDS.append(int(threshold))

    print(f'Quintile threshold for defect type {x} = {int(threshold)}')
THRESHOLDS[3] = 3500
!pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null

package_path = '../input/unetmodelscript'

import sys

sys.path.append(package_path)
import pdb

import cv2

import torch

from tqdm import tqdm

import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, Dataset

from albumentations import (Normalize, Compose)

from albumentations.pytorch import ToTensor

import torch.utils.data as data

from model import Unet
#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



class TestDataset(Dataset):

    '''Dataset for test prediction'''

    def __init__(self, root, df, mean, std):

        self.root = root

        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

        self.fnames = df['ImageId'].unique().tolist()

        self.num_samples = len(self.fnames)

        self.transform = Compose(

            [

                Normalize(mean=mean, std=std, p=1),

                ToTensor(),

            ]

        )



    def __getitem__(self, idx):

        fname = self.fnames[idx]

        path = os.path.join(self.root, fname)

        image = cv2.imread(path)

        images = self.transform(image=image)["image"]

        return fname, images



    def __len__(self):

        return self.num_samples
def post_process(probability, threshold, min_size):

    '''Post processing of each predicted mask, components with lesser number of pixels

    than `min_size` are ignored'''

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((256, 1600), np.float32)

    num = 0

    for c in range(1, num_component):

        p = (component == c)

        if p.sum() > min_size:

            predictions[p] = 1

            num += 1

    return predictions, num



sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'

test_data_folder = "../input/severstal-steel-defect-detection/test_images"



# initialize test dataloader

best_threshold = 0.5

num_workers = 2

batch_size = 4

min_size = 0 #we will use the 3500 threshold later

mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)

df = pd.read_csv(sample_submission_path)

testset = DataLoader(

    TestDataset(test_data_folder, df, mean, std),

    batch_size=batch_size,

    shuffle=False,

    num_workers=num_workers,

    pin_memory=True

)



# Initialize mode and load trained weights

ckpt_path = "../input/unetstartermodelfile/model.pth"

device = torch.device("cuda")

model = Unet("resnet18", encoder_weights=None, classes=4, activation=None)

model.to(device)

model.eval()

state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

model.load_state_dict(state["state_dict"])



# start prediction

predictions = []

for i, batch in enumerate(testset):

    fnames, images = batch

    batch_preds = torch.sigmoid(model(images.to(device)))

    batch_preds = batch_preds.detach().cpu().numpy()

    for fname, preds in zip(fnames, batch_preds):

        for cls, pred in enumerate(preds):

            pred, num = post_process(pred, best_threshold, min_size)

            rle = mask2rle(pred)

            name = fname + f"_{cls+1}"

            predictions.append([name, rle])



# save predictions to submission.csv

sub_init = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
sub_init['ClassId'] = sub_init['ImageId_ClassId'].apply(lambda x: x.split('_')[1]).astype('int8')



DEFECT_TYPES = [1, 2, 3, 4]



for x in DEFECT_TYPES:

    sub_init.loc[sub_init.ClassId==x, 'pixel_count'] = sub_init.loc[sub_init.ClassId==x, 'EncodedPixels'].apply(pixel_count) 



for i, x in enumerate(DEFECT_TYPES):

    try:

        df = sub_init.loc[sub_init.ClassId==x, :]

        ax = df['pixel_count'].plot(kind='hist', bins=100, figsize=(15, 6))

        df['pixel_count'].plot(kind='kde', ax=ax, secondary_y=True)

        plt.xlim(left=0)

        plt.axvline(x=3500, linewidth=2, color='r')

        plt.axvline(x=THRESHOLDS[i], linewidth=2, color='g')

        plt.title(f'Pixel Distribution for Defect Type {x}')

        print(f'=====\nDEFECT TYPE {x}\n=====')

        print(df['pixel_count'].describe())

        plt.show()

    except:

        pass
INIT_THRESHOLD = np.full(4, 3500)
sub_1 = sub_init.copy()

sub_2 = sub_init.copy()



for i in range(len(THRESHOLDS)):

    threshold_init = INIT_THRESHOLD[i]

    threshold_new = THRESHOLDS[i]

    defect = DEFECT_TYPES[i]

    

    sub_1.loc[(sub_1['ClassId']==defect) & (sub_1.pixel_count < threshold_init), 'EncodedPixels'] = ''

    sub_2.loc[(sub_2['ClassId']==defect) & (sub_2.pixel_count < threshold_new), 'EncodedPixels'] = ''

    

sub_1.drop(['ClassId', 'pixel_count'], axis=1, inplace=True)    

sub_2.drop(['ClassId', 'pixel_count'], axis=1, inplace=True)    
initial_sub_len =  len(sub_1.loc[sub_1['EncodedPixels'] != '', :])

modified_sub_len = len(sub_2.loc[sub_2['EncodedPixels'] != '', :])

print(f'Positive instance case count with 3500 pixel threshold: {initial_sub_len}')

print(f'Positive instance case count with new pixel thresholds: {modified_sub_len}')
#original submission

#sub_1.to_csv('submission.csv', index=False)



#submission with optimised pixel thresholds

sub_2.to_csv('submission.csv', index=False)
def pixel_postprocess(sub, thresholds):

    

    '''

    sub: submission dataframe that includes model predictions

    thresholds: list or array of 4 sequential integers for the minimum number of pixels per defect type 

    '''

    

    #define function for creating count of pixels 

    def pixel_count(rle):

        

        if np.logical_or(rle==0, rle==''):

            return(np.nan)

        array = np.asarray([int(x) for x in rle.split()])

        lengths = array[1::2]

    

        return sum(lengths)

    

    

    #this stage used to simplify subsetting by defect type

    DEFECT_TYPES = [1, 2, 3, 4]

    sub['ClassId'] = sub['ImageId_ClassId'].apply(lambda x: x.split('_')[1]).astype('int8')

    

    #pixels counts for each positive case

    for x in DEFECT_TYPES:

        sub.loc[sub.ClassId==x, 'pixel_count'] = sub.loc[sub.ClassId==x, 'EncodedPixels'].apply(pixel_count) 

    

    #for each defect type and its associated threshold, remove positive instances with pixel_count < threshold

    for i in range(len(thresholds)):

        threshold = thresholds[i]

        defect = DEFECT_TYPES[i]

        sub.loc[(sub['ClassId']==defect) & (sub.pixel_count < threshold), 'EncodedPixels'] = ''

    

    #remove pixel count and ClassId columns so submission has correct fields

    sub.drop(['ClassId', 'pixel_count'], axis=1, inplace=True)

    

    #don't forget to title it "submission.csv" when uploading!

    return(sub)    