import os



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



import cv2

import skimage.io



from sklearn.model_selection import train_test_split, StratifiedKFold



import torch

import torch.nn as nn

from torch.utils.data import DataLoader, Dataset



import torchvision

from torchvision import models, transforms



from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip

from albumentations.pytorch import ToTensorV2



import torch.optim as optim

from torch.optim import lr_scheduler



import time

import copy



from PIL import Image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
train = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")

test = pd.read_csv("../input/prostate-cancer-grade-assessment/test.csv")

sample = pd.read_csv("../input/prostate-cancer-grade-assessment/sample_submission.csv")
sample.head()
def tile(img, sz=128, N=16):

    shape = img.shape

    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz

    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],

                 constant_values=255)

    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)

    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

    if len(img) < N:

        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]

    img = img[idxs]

    return img
# def get_transforms(*, data):

    

#     assert data in ('train', 'valid')

    

#     if data == 'valid':

#         return Compose([

#             Normalize(

#                 mean=[0.485, 0.456, 0.406],

#                 std=[0.229, 0.224, 0.225],

#             ),

#             ToTensorV2(),

#         ])



def get_transforms(*, data):

    

    assert data in ('train', 'valid')

    

    if data == 'valid':

        return transforms.Compose([transforms.ToTensor(),

                                   transforms.Normalize(

                                       mean=[0.485, 0.456, 0.406],

                                       std=[0.229, 0.224, 0.225])

                                  ]

                                 )
class TestDataset(Dataset):

    def __init__ (self, image_id, transform=None):

        self.image_id = image_id

#         self.dir_name = dir_name

        self.transform = transform

        

    def __len__(self):

        return len(self.image_id)

    

    def __getitem__(self, idx):

#         Creating the image

        name = self.image_id[idx]

#         file_path = f"../input/prostate-cancer-grade-assessment/test_images/{name}.tiff"

        file_path = f"../input/prostate-cancer-grade-assessment/test_images/{name}.tiff"

        image = skimage.io.MultiImage(file_path)[-1]

        image = tile(image, sz=128, N=16)

        image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]),

                                   cv2.vconcat([image[4], image[5], image[6], image[7]]),

                                   cv2.vconcat([image[8], image[9], image[10], image[11]]),

                                   cv2.vconcat([image[12], image[13], image[14], image[15]])])

#         image_tiles = cv2.cvtColor(image_tiles, cv2.COLOR_BGR2RGB)

        if self.transform:

            augmented = self.transform(image=image)

            image = augmented['image']

#         Return image

        return image
# Creates probabilites of predictions

def inference(model, test_loader, device):

#     send model to gpu

    model.to(device)

    predictions = []

    

    for i, images in enumerate(test_loader):

#         send images to gpu

        images = images.to(device)

#         Temporary sets the require_gradient parameter to False.

#         Pytorch doesn't have to calcuate the gradients of the weights

        with torch.no_grad():

            outputs = model(images)

            

        predictions.append(outputs.to('cpu').numpy().argmax(1))

#         Send predictions back to cpu (save memory) convert to numpy

    

    predictions = np.concatenate(predictions)

#     Joins the arrays into one axis

    return predictions
if os.path.exists("../input/prostate-cancer-grade-assessment/test_images"):

    print('test_images_exist')

else:

    print('test_images_not_exist')
model = torch.load("../input/complete-model/complete_model.pth")

model.fc

model.to(device)
def submit(sample):

    if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

        test_dataset = TestDataset(sample['image_id'], transform=get_transforms(data='valid'))

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        preds = inference(model, test_loader, device)

        sample['isup_grade'] = preds

    return sample
submission = submit(sample)

submission['isup_grade'] = submission['isup_grade'].astype(int)

submission.head()
submission.to_csv('submission.csv', index=False)