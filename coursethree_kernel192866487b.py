import os
import cv2
from PIL import Image
import time
import copy
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from albumentations import (HorizontalFlip,VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise,RandomRotate90,Transpose,RandomBrightnessContrast,RandomCrop)
from albumentations.pytorch import ToTensor
import albumentations as albu
import matplotlib.image as mpi
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

DIR_INPUT = '/kaggle/input/agedetection'
DIR_TRAIN = f'{DIR_INPUT}/train_DETg9GD'
DIR_TEST = f'{DIR_INPUT}/test_Bh8pGW3'
path = f'{DIR_TEST}/Test/10.jpg'
img = plt.imread(path)
plt.imshow(img)
print(img.shape)
test_df = pd.read_csv(f'{DIR_TEST}/test.csv')
test_df.head()
