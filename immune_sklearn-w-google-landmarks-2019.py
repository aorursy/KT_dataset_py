# 🎨 Justin Faler



import os

import time

import multiprocessing



from os.path import isfile, join, basename, splitext, isfile, exists

from typing import Any, Optional, Tuple



import sklearn as sk

import pandas as pd

import numpy as np



import torch

import torch.nn as nn

import torchvision

import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn

from torch.utils.data import TensorDataset, DataLoader, Dataset

from sklearn.preprocessing import LabelEncoder

from PIL import Image

from tqdm import tqdm



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns
print(os.listdir('../input'))
import pandas as pd

df = pd.read_csv('../input/train.csv')
# Print the shape 

df.shape
# Print the size

df.size
# This will print the dataframe 🖼️

df
df.loc[0]