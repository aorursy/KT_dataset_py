import os
import pandas as pd
import numpy as np
import json
import warnings
import random
import torch
import shutil
from tqdm import tqdm
from pathlib import Path
warnings.filterwarnings('ignore')
torch.cuda.is_available()
os.listdir('/kaggle/input/blood-cell-count-and-typesdetection')
def seed_system(seed=4):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def flush_folders(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory) 
    return True

seed = seed_system()
flush = flush_folders(directory='/kaggle/working/blood-cell-count-and-typesdetection')
# os.listdir()
# !git clone https://github.com/ultralytics/yolov5 # clone repo 
# !curl -L -o tmp.zip https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip && unzip -q tmp.zip && rm tmp.zip # download dataset 
# !pip install -qr yolov5/requirements.txt # install dependencies 
make_folders = ['/images','/labels']
train_valid = ['/train','/valid']
working_dir = '/kaggle/working/blood-cell-count-and-typesdetection'


def create_folder_structure(home=working_dir,make_folders=make_folders,train_valid=train_valid):
    if not os.path.exists(home):
        os.makedirs(home)
    for path in tqdm(make_folders):
        for c in train_valid:
            data_path = home+path+c
            if not os.path.exists(data_path):
                os.makedirs(data_path)
    return True
create_folder_structure()
# !cp -r /kaggle/input/blood-cell-count-and-typesdetection /kaggle/working
os.listdir('blood-cell-count-and-typesdetection/images')
#os.listdir('/kaggle/input/blood-cell-count-and-typesdetection/images/images/')
!cp -r /kaggle/input/blood-cell-count-and-typesdetection/images/images/ /kaggle/working/blood-cell-count-and-typesdetection/
!cp -r /kaggle/input/blood-cell-count-and-typesdetection/labels/labels/ /kaggle/working/blood-cell-count-and-typesdetection/
!cp /kaggle/input/blood-cell-count-and-typesdetection/bcc-kaggle.yaml /kaggle/working/yolov5/bcc-kaggle.yaml
import yaml

with open("/kaggle/working/blood-cell-count-and-typesdetection//bcc-kaggle.yaml") as f:
     list_doc = yaml.load(f)

list_doc['train'] = '/kaggle/working/blood-cell-count-and-typesdetection/images/train'
list_doc['val'] = '/kaggle/working/blood-cell-count-and-typesdetection/images/valid'

with open("/kaggle/working/yolov5/bcc-kaggle.yaml", "w") as f:
    yaml.dump(list_doc, f)
list_doc
!python yolov5/train.py --img 128 --batch 16 --epochs 100 --data /kaggle/working/yolov5/bcc-kaggle.yaml --cfg /kaggle/working/yolov5/models/yolov5s.yaml --name bcc
from PIL import Image
os.listdir('runs/exp14_bcc/weights')
# from utils.utils import plot_results; plot_results()  # plot results.txt as results.png
Image.open('runs/exp14_bcc/results.png') 
!python yolov5/detect.py --source /kaggle/working/blood-cell-count-and-typesdetection/images/valid/ --weights runs/exp14_bcc/weights/best.pt
os.listdir('inference/output')
Image.open('inference/output/BloodImage_00253.jpg')