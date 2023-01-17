!pip install -q efficientnet_pytorch > /dev/null
from glob import glob

from tqdm import tqdm

import pandas as pd

from sklearn.model_selection import GroupKFold

import cv2

from skimage import io

import albumentations as A

import scipy as sp

import torch

import os

from datetime import datetime

import time

import random

import cv2

import pandas as pd

import numpy as np

import albumentations as A

import matplotlib.pyplot as plt

from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset,DataLoader

from torch.utils.data.sampler import SequentialSampler, RandomSampler

from torch.nn import functional as F

from glob import glob

import sklearn

from torch import nn

import warnings



warnings.filterwarnings("ignore") 

warnings.filterwarnings("ignore", category=DeprecationWarning) 



SEED = 42



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)
DATA_PATH = '../input/melanoma-merged-external-data-512x512-jpeg'
TEST_ROOT_PATH = f'{DATA_PATH}/512x512-test/512x512-test'



def get_valid_transforms():

    return A.Compose([

            A.Resize(height=512, width=512, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)



class DatasetRetriever(Dataset):



    def __init__(self, image_ids, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.transforms = transforms



    def __getitem__(self, idx: int):

        image_id = self.image_ids[idx]

        image = cv2.imread(f'{TEST_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = image.astype(np.float32) / 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']

        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
from efficientnet_pytorch import EfficientNet



def get_net():

    net = EfficientNet.from_name('efficientnet-b5')

    net._fc = nn.Linear(in_features=2048, out_features=2, bias=True)

    return net



net = get_net().cuda()
df_test = pd.read_csv(f'../input/siim-isic-melanoma-classification/test.csv', index_col='image_name')



test_dataset = DatasetRetriever(

    image_ids=df_test.index.values,

    transforms=get_valid_transforms(),

)



test_loader = torch.utils.data.DataLoader(

    test_dataset, 

    batch_size=8,

    num_workers=2,

    shuffle=False,

    sampler=SequentialSampler(test_dataset),

    pin_memory=False,

    drop_last=False,

)
checkpoint_path = '../input/melanoma-public-checkpoints/effnet5-best-score-checkpoint-015epoch-version2.bin'

checkpoint = torch.load(checkpoint_path)

net.load_state_dict(checkpoint);

net.eval();
result = {'image_name': [], 'target': []}

for images, image_names in tqdm(test_loader, total=len(test_loader)):

    with torch.no_grad():

        images = images.cuda().float()

        outputs = net(images)

        y_pred = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,1]



    result['image_name'].extend(image_names)

    result['target'].extend(y_pred)



submission = pd.DataFrame(result)
submission.to_csv('submission.csv', index=False)

submission['target'].hist(bins=100);