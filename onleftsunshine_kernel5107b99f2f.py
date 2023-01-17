import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

import torch.backends.cudnn as cudnn



import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

import pandas as pd

import time

import csv

from pathlib import Path

import zipfile,os



zipFile = zipfile.ZipFile('test1.zip', 'w')

zipFile.write('./model_statistics','model',zipfile. ZIP_STORED)
