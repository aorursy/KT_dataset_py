import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import os

from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/thai-mnist-classification/mnist.train.map.csv").set_index("id")

df
# drop unwanted files

# drop = pd.read_csv("mydrop.csv").set_index("id")

# df = df.drop(drop.index, axis=0)
x_train, x_val, y_train, y_val = train_test_split(df.index,df["category"].values, test_size = 0.2, random_state=42)
y_train

classes = pd.DataFrame({'freq': y_train})

classes = classes.groupby('freq', as_index=False)

classes.size()['size'].plot(kind='bar')

plt.show()
denominator = 0

percent = 0

weight_list = []

print("percentage of each class")

for n,i in enumerate(classes.size()['size']):

    percent = (i*100/6435)

    numerator = 1/percent

    denominator += numerator

    weight_list.append(numerator)

    print("class {}: {} %".format(n,percent))

    print("weight {}".format(numerator))

print("total weight",denominator)
adjusted = []

for i in range(len(classes.size())):

    a = classes.size()['size'][i]*weight_list[i]

    adjusted.append(a)

adjusted = pd.Series(adjusted)

adjusted.plot(kind='bar')
#อย่าลืม apply weight list -- eg. criterion = torch.nn.NLLLoss(weight = torch.FloatTensor(weight_list).to(device)) ต้องทำเป็น tensor + ย้ายเข้า GPU ก่อนถึงใช้ได้ครับ
from PIL import Image, ImageChops, ImageOps

import torch

from torchvision import datasets

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import cv2

from PIL import ImageFilter

from PIL.ImageFilter import (BLUR, SHARPEN)
def trim(im):

    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))

    diff = ImageChops.difference(im, bg)

    bbox = diff.getbbox()

    if bbox:

        return im.crop(bbox)
im_size = 64

batch_size = 128



class Dataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transforms, augment=True):

        self.x = x

        self.y = y

        self.transforms = transforms

        self.augment = augment

    def __len__(self):

        return len(self.x)

    def __getitem__(self, index):

        label = self.y[index]

        ID = '../input/thai-mnist-classification/train/' + self.x[index]

        img = Image.open(ID).convert('L')

        

        #trim(and automatically center), add border(เพื่อให้ใช้ zoom ได้)

        img = trim(img)

        img = ImageOps.expand(img, border=60, fill="white")

        

        """ augment """

        if self.augment==True:

            #BLUR ทำให้เส้นหนา ->binarize -> resize(with antialiasing ในตัว)

            #Blur

            img = img.filter(BLUR)

            img = img.filter(BLUR)

            #to open cv and binarize

            img = np.array(img)

            _,img = cv2.threshold(img, 254.999, 255, cv2.THRESH_BINARY)

            #back to PIL and invert

            img = Image.fromarray(img)

            img = ImageOps.invert(img)

        else:

            img = ImageOps.invert(img)

            

        #Resize(+antialias), convert to tensor(0-1), normalize

        if self.transforms is not None:

            img = self.transforms(img)

        

        return img, label



transformations = transforms.Compose([

                                    transforms.Resize((im_size,im_size)), #will also antialias

                                    transforms.ToTensor(), #to tensor will convert 0-255->0-1

                                    transforms.Normalize((0.5), (0.5))

                                    ])







train_dataset = Dataset(x=x_train, y=y_train, transforms=transformations)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False)

No_augment = Dataset(x=x_train, y=y_train, transforms=transformations, augment=False)

No_augment_loader = DataLoader(No_augment, batch_size = batch_size, shuffle=False)

# val_dataset = Dataset(x=x_val, y=y_val, transforms=transformations)

# val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True)
dataiter = iter(No_augment_loader)

images, labels = dataiter.next()

print(images.shape)

img = images.numpy()

fig = plt.figure(figsize=(20, 4))

fig.suptitle('Before', fontsize=20)

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.squeeze(images[idx]), cmap="gray")
dataiter = iter(train_loader)

images, labels = dataiter.next()

print(images.shape)

img = images.numpy()

fig = plt.figure(figsize=(20, 4))

fig.suptitle('After', fontsize=20)

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.squeeze(images[idx]), cmap="gray")