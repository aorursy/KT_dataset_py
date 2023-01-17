# Import packages
!pip install -q efficientnet_pytorch > /dev/null
from glob import glob
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import torch
from torch import nn
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn
# Original
def get_test_transforms1():
    return A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)

# HorizontalFlip
def get_test_transforms2():
    return A.Compose([
        A.HorizontalFlip(p=1),
        A.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)

# VerticalFlip
def get_test_transforms3():
    return A.Compose([
        A.VerticalFlip(p=1),
        A.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)

# HorizontalFlip & VerticalFlip
def get_test_transforms4():
    return A.Compose([
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)

DATA_ROOT_PATH = '../input/alaska2-image-steganalysis'

from efficientnet_pytorch import EfficientNet


def get_net():
    net = EfficientNet.from_pretrained('efficientnet-b2')
    net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
    return net


net = get_net().cuda()

checkpoint = torch.load('../input/model33/best-checkpoint-033epoch.bin')
net.load_state_dict(checkpoint['model_state_dict'])
class DatasetSubmissionRetriever(Dataset):

    def __init__(self, image_names, transforms1, transforms2, transforms3, transforms4):
        super().__init__()
        self.image_names = image_names
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.transforms3 = transforms3
        self.transforms4 = transforms4

    def __getitem__(self, index: int):
        image_name = self.image_names[index]

        image1 = cv2.imread(f'{DATA_ROOT_PATH}/Test/{image_name}', cv2.IMREAD_COLOR)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB).astype(np.float32)
        image1 /= 255.0

        sample = {'image': image1}
        sample1 = self.transforms1(**sample)
        image1 = sample1['image']

        sample2 = self.transforms2(**sample)
        image2 = sample2['image']

        sample3 = self.transforms3(**sample)
        image3 = sample3['image']

        sample4 = self.transforms4(**sample)
        image4 = sample4['image']
        return image_name, image1, image2, image3, image4

    def __len__(self) -> int:
        return self.image_names.shape[0]


dataset = DatasetSubmissionRetriever(
    image_names=np.array([path.split('/')[-1] for path in glob('../input/alaska2-image-steganalysis/Test/*.jpg')]),
    transforms1=get_test_transforms1(),
    transforms2=get_test_transforms2(),
    transforms3=get_test_transforms3(),
    transforms4=get_test_transforms4(),
)

data_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)

result = {'Id': [], 'Label': []}
for step, (image_names, images1, images2, images3, images4) in enumerate(data_loader):
    print(step, end='\r')

    y_pred1 = net(images1.cuda())
    y_pred1 = 1 - nn.functional.softmax(y_pred1, dim=1).data.cpu().numpy()[:, 0]

    y_pred2 = net(images2.cuda())
    y_pred2 = 1 - nn.functional.softmax(y_pred2, dim=1).data.cpu().numpy()[:, 0]

    y_pred3 = net(images3.cuda())
    y_pred3 = 1 - nn.functional.softmax(y_pred3, dim=1).data.cpu().numpy()[:, 0]

    y_pred4 = net(images4.cuda())
    y_pred4 = 1 - nn.functional.softmax(y_pred4, dim=1).data.cpu().numpy()[:, 0]

    result['Id'].extend(image_names)
    # The weights can change by yourself
    result['Label'].extend((y_pred1 * 3 + y_pred2 + y_pred3 + y_pred4) / 6)

submission = pd.DataFrame(result)
submission.to_csv('submission.csv', index=False)
print(submission.head())