import torch, torchvision

from pathlib import Path

import numpy as np

import cv2

import pandas as pd

from tqdm import tqdm

import PIL.Image as Image

import seaborn as sns

from pylab import rcParams

import matplotlib.pyplot as plt

from matplotlib import rc

from matplotlib.ticker import MaxNLocator

from torch.optim import lr_scheduler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from glob import glob

import shutil

from collections import defaultdict

from torch import nn, optim

import torch.nn.functional as F

import torchvision.transforms as T

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

from torchvision import models

%matplotlib inline

%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
!wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip

!unzip -qq GTSRB_Final_Training_Images.zip