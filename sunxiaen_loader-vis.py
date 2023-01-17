# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print()
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
!pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
coco_det=datasets.CocoDetection(root='../input/coco2017/train2017/train2017',annFile='../input/coco2017/annotations_trainval2017/annotations/instances_train2017.json',transform=transforms.ToTensor())
def collate_fn_coco(batch):
    return tuple(zip(*batch))

sampler = torch.utils.data.RandomSampler(coco_det)
batch_sampler = torch.utils.data.BatchSampler(sampler, 8, drop_last=True)
 
# 创建 dataloader
data_loader = torch.utils.data.DataLoader(
        coco_det, batch_sampler=batch_sampler, num_workers=0,
        collate_fn=collate_fn_coco)
next(iter(data_loader))
import cv2
import random
import matplotlib.pyplot as plt
%matplotlib inline
font = cv2.FONT_HERSHEY_SIMPLEX
for imgs,labels in data_loader:
    for i in range(len(imgs)):
        bboxes = []
        ids = []
        img = imgs[i]
        labels_ = labels[i]
        for label in labels_:
            bboxes.append([label['bbox'][0],
            label['bbox'][1],
            label['bbox'][0] + label['bbox'][2],
            label['bbox'][1] + label['bbox'][3]
            ])
            ids.append(label['category_id'])
 
        img = img.permute(1,2,0).numpy()
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        for box ,id_ in zip(bboxes,ids):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            print(x1,y1,x2,y2)
            print(img.shape)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),thickness=2)
            cv2.putText(img, text=str(id_), org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1, 
                thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0))
        plt.imshow(img)
        plt.show()
#         cv2.waitKey()
