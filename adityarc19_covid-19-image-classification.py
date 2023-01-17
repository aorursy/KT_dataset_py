import numpy as np

import pandas as pd 



import os

import torch

import torchvision

from torch.utils.data import random_split



from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor
data_dir = '../input/covid19-image-dataset/Covid19-dataset'



print(os.listdir(data_dir))

classes = os.listdir(data_dir + "/train")

print(classes)
dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
print(dataset.classes)
img, label = dataset[0]

print(img.shape, label)

img
import matplotlib.pyplot as plt



def show_example(img, label):

    print('Label: ', dataset.classes[label], "("+str(label)+")")

    plt.imshow(img.permute(1, 2, 0))
show_example(*dataset[19])