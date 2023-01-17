import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import transforms,models

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import random

from PIL import Image

import os

%matplotlib inline
os.listdir("../input")
os.listdir("../input/checkpoint50")
checkpoint = torch.load("../input/checkpoint50/checkpoint_epoch_50.pt",map_location='cpu')
checkpoint
model = checkpoint['model']
class TestDataset(torch.utils.data.Dataset):

    

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir

        self.transform = transform

        self.filenames = os.listdir(self.root_dir)

        

    def __len__(self):

        return len(self.filenames)

    

    def __getitem__(self, idx):

        path = "{}/{}".format(self.root_dir, self.filenames[idx])

        image = Image.open(path)

        if image.getbands()[0] == 'L':

            image = image.convert('RGB')

        return (self.transform(image), self.filenames[idx])
test_data_dir = "../input/aptos2019-blindness-detection/test_images"

test_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

])



test_dataset = TestDataset(test_data_dir, test_transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False)
model.eval()
id_codes = []

diags = []



for imgs,files in test_loader:

    logpbs = model(imgs)

    preds = torch.exp(logpbs)

    _ , diagnosis = torch.max(preds, 1)

    for id, diag in zip(files, diagnosis):

        id_codes.append(id.replace(".png",""))

        diags.append(diag.item())

        

df = pd.DataFrame({"id_code" : id_codes, "diagnosis" : diags})

df.to_csv("./submission.csv", index=False)
df.head()