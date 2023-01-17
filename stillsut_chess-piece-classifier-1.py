!pip install git+https://github.com/fastai/fastcore > /dev/null

!pip install git+https://github.com/fastai/fastai2 > /dev/null
import os, sys

import json

import numpy as np

import pandas as pd



import torch

import torchvision



from voc2coco import main_conv

from cropper import main_crop
from  fastai2.vision.all import *
input_dir = '/kaggle/input/chess-piece-images-and-bounding-boxes/export/'

output_dir = '/kaggle/working/'



main_conv(input_dir, output_dir, b_summary=True)



main_crop(img_dir=input_dir)
img_dir = './crops/'

fns = os.listdir(img_dir)

print(f"num image files: {len(fns)}")

fns[:5]