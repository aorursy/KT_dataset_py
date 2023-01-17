#Usual Imports

import skimage.io

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import PIL

import cv2

import torch

from pathlib import Path

import pandas as pd

from tqdm import tqdm
#Check input file

!ls ../input/prostate-cancer-grade-assessment/train_images/008069b542b0439ed69b194674051964.tiff
#Load input file and check dimensions

file_path = f'../input/prostate-cancer-grade-assessment/train_images/008069b542b0439ed69b194674051964.tiff'

image = skimage.io.MultiImage(file_path)

image = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)

image.shape
#Display the file

plt.imshow(image)
#Convert to torch and CHW format

input = torch.from_numpy(image)

input.transpose_(0, 2).shape
# Create patches of size 512x512

patch_size = 512

stride=patch_size



patches = input.data.unfold(0, 3, 3).unfold(1, patch_size, stride).unfold(2, patch_size, stride)

patches.shape
def plot_image(tensor):

    plt.figure()

    plt.imshow(tensor.numpy().transpose(1, 2, 0))

    plt.show()

    

def showTensor(aTensor):

    plt.figure()

    plt.imshow(aTensor.numpy())

    plt.colorbar()

    plt.show()
#Load mask file

mask_path = f'../input/prostate-cancer-grade-assessment/train_label_masks/008069b542b0439ed69b194674051964_mask.tiff'

mask = skimage.io.MultiImage(mask_path)

mask = cv2.cvtColor(mask[0], cv2.COLOR_BGR2RGB)

mask.shape
#Only the third channel has the markings

np.unique(mask[:,:,2])
#Convert to torch and CHW format

input_mask = torch.from_numpy(mask)

input_mask.transpose_(0, 2).shape
mask_patches = input_mask.data.unfold(0, 3, 3).unfold(1, patch_size, stride).unfold(2, patch_size, stride)

mask_patches.shape
#Show a Random Patch

plot_image(patches[0][40][3])

showTensor(mask_patches[0][40][3][-1]);
train=pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

check_score = train.loc[train['data_provider'] == 'radboud',['image_id','gleason_score']]
check_score=check_score.head(5)
output=list()

for index, row in tqdm(check_score.iterrows()):

    filename = row.image_id

    gleason_score = row.gleason_score

    mask_path = Path('../input/prostate-cancer-grade-assessment/train_label_masks/'+filename+'_mask.tiff')

    mask = skimage.io.MultiImage(str(mask_path))

    mask = cv2.cvtColor(mask[0], cv2.COLOR_BGR2RGB)

    input_mask = torch.from_numpy(mask)

    input_mask.transpose_(0, 2)



    a,b = torch.unique(input_mask, return_counts=True)

    a,b = a.numpy(),b.numpy()

    i = 0 if a[0]!=0 else 1    

    c = b[i:]/np.sum(b[i:])*100



    #dict(zip(a[1:], c))

    final = [filename,[(k,v) for k,v in dict(zip(a[1:], c)).items()],gleason_score]

    del mask_path,mask,input_mask, a,b,i,c 

    output = output+final

    
output