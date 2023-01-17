# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

DIR_INPUT = '//kaggle/input/siim-isic-melanoma-classification'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

DIR_WEIGHTS = '/kaggle/input/siim-isic-melanoma-classification'

WEIGHTS_FILE = '/kaggle/input/faster11/fasterrcnn_resnet50_fpn-11.pth'
test_meta = pd.read_csv(f'{DIR_INPUT}/test.csv')
image_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/test"
test_meta

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

class TrainData(Dataset):

    def __init__(self, dataframe, image_dir, transforms):
        super().__init__()
        
        self.df = dataframe
        self.image_ids = dataframe['image_name'].unique()
        self.image_dir = image_dir
        self.transforms = transforms


    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image.astype(np.float32)/ 255
        
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
            
        return image, image_id
    
    def __len__(self) -> int:
        return self.image_ids.shape[0]
test_dataset = TrainData(test_meta, image_dir, transforms = get_valid_transforms())
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 0)
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import torch

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/kaggle/input/siimmodel/model.pth"
model = torch.load(model_path)
model.to(device)
model.eval()
from tqdm import tqdm

result = {'image_name': [], 'target': []}
for images, image_names in tqdm(test_loader):
    with torch.no_grad():
        images = images.cuda().float()
        outputs = model(images)
        y_pred = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,1]

    result['image_name'].extend(image_names)
    result['target'].extend(y_pred)

submission = pd.DataFrame(result)

print(submission)
submission.to_csv('submission.csv', index=False)
