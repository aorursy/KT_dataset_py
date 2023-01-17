%mkdir /kaggle/temp
%cd /kaggle/temp
!git clone https://github.com/ultralytics/yolov5  # clone repo
#!pip install -r yolov5/requirements.txt  # install dependencies (ignore errors)
%cd yolov5

import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
from distutils.dir_util import copy_tree
import os

temp_input = '/kaggle/temp/input/'

image_path = os.path.join(temp_input, 'images')
copy_tree('/kaggle/input/synthetic-gloomhaven-monsters/images', image_path)

label_path = os.path.join(temp_input, 'labels')
copy_tree('/kaggle/input/synthetic-gloomhaven-monsters/labels', label_path)
%cp /kaggle/input/synthetic-gloomhaven-monsters/*.yaml ./
# Start tensorboard (optional)
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/yolov5x/runs
!python train.py --batch 4 --epochs 100 --cfg "yolov5x.yaml" --data "coco.yaml" --weight yolov5x.pt --logdir "/kaggle/working/yolov5x/runs"