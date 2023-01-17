###### This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install albumentations

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# libraries

import os

import cv2

import gc

import matplotlib.pyplot as plt

import random



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import roc_auc_score

import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.models as models

import torchvision.transforms as transforms

import torch.optim as optim

from torch.optim import lr_scheduler

import time 

import tqdm

from PIL import Image

train_on_gpu = True

from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR



import albumentations

from albumentations import torch as AT



n_splits = 5

batch_size = 32

n_epochs = 5

patience = 15

SEED = 323

num_workers = 0

num_tta = 64



def seed_everything(seed=SEED):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()
labels = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')

tr, val = train_test_split(labels.label, stratify=labels.label, test_size=0.07, random_state=SEED)

img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}



class CancerDataset(Dataset):

    def __init__(self, datafolder, datatype='train', idx=[], transform = transforms.Compose([transforms.CenterCrop(48),transforms.ToTensor()]), labels_dict={}):

        self.datafolder = datafolder

        self.datatype = datatype

        self.image_files_list = [s for s in os.listdir(datafolder)]

        self.image_files_list = [self.image_files_list[i] for i in idx]

        self.transform = transform

        self.labels_dict = labels_dict

        if self.datatype == 'train':

            self.labels = [labels_dict[i.split('.')[0]] for i in self.image_files_list]

        else:

            self.labels = [0 for _ in range(len(self.image_files_list))]



    def __len__(self):

        return len(self.image_files_list)



    def __getitem__(self, idx):

        img_name = os.path.join(self.datafolder, self.image_files_list[idx])

        img = cv2.imread(img_name)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = self.transform(image=img)

        image = image['image']



        img_name_short = self.image_files_list[idx].split('.')[0]



        if self.datatype == 'train':

            label = self.labels_dict[img_name_short]

        else:

            label = 0

        return image, label





data_transforms = albumentations.Compose([

    albumentations.Resize(224, 224),

    albumentations.RandomRotate90(p=0.5),

    albumentations.Transpose(p=0.5),

    albumentations.Flip(p=0.5),

    albumentations.OneOf([

        albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(), albumentations.IAAEmboss(), 

        albumentations.RandomBrightness(), albumentations.RandomContrast(),

        albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5), 

    albumentations.HueSaturationValue(p=0.5), 

    albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),

    albumentations.Normalize(),

    AT.ToTensor()

    ])

data_transforms_test = albumentations.Compose([

    albumentations.Resize(224, 224),

    albumentations.Normalize(),

    AT.ToTensor()

    ])





class Densenet169(nn.Module):

    def __init__(self, pretrained=True):

        super(Densenet169, self).__init__()

        self.model = models.densenet169(pretrained=pretrained)

        self.linear = nn.Linear(1000+2, 16)

        self.bn = nn.BatchNorm1d(16)

        self.dropout = nn.Dropout(0.2)

        self.elu = nn.ELU()

        self.out = nn.Linear(16, 1)

    

    def forward(self, x):

        out = self.model(x)

        batch = out.shape[0]

        max_pool, _ = torch.max(out, 1, keepdim=True)

        avg_pool = torch.mean(out, 1, keepdim=True)



        out = out.view(batch, -1)

        conc = torch.cat((out, max_pool, avg_pool), 1)



        conc = self.linear(conc)

        conc = self.elu(conc)

        conc = self.bn(conc)

        conc = self.dropout(conc)



        res = self.out(conc)



        return res





model_conv = Densenet169(pretrained=False)

model_conv.load_state_dict(torch.load("../input/densenet169-pretrain-cancer/model"), strict=False)

model_conv.eval().cuda()
for tta in range(num_tta):

    

    seed_everything(SEED+10*tta)

    test_idx = [i for i in range(len(os.listdir("../input/histopathologic-cancer-detection/test")))]

    test_set = CancerDataset(datafolder='../input/histopathologic-cancer-detection/test/', idx=test_idx, datatype='test', transform=data_transforms)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    preds = []

    for batch_i, (data, target) in enumerate(test_loader):

        data, target = data.cuda(), target.cuda()

        output = model_conv(data).detach()



        pr = output[:,0].cpu().numpy()

        for i in pr:

            preds.append(i)

        

    test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})

    test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])

    sub = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

    sub = pd.merge(sub, test_preds, left_on='id', right_on='imgs')

    sub = sub[['id', 'preds']]

    sub.columns = ['id', 'label']

    sub.head()

    sub.to_csv('single_model_'+str(tta)+'.csv', index=False)



del model_conv

gc.collect()

torch.cuda.empty_cache()
def sigmoid(x):

  return 1 / (1 + np.exp(-x))



def sigmoid_pd(df):

    df["label"] = df["label"].apply(sigmoid)

    return df
df = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')



for tta in range(num_tta):

    df0 = pd.read_csv('../working/single_model_'+str(tta)+'.csv')

    df0 = sigmoid_pd(df0)

    df['label'] += df0['label']

    if(tta+1==8):

        df_tmp = df.copy()

        df_tmp['label'] /= 8 

        df_tmp.to_csv('submission_tta_'+str(tta+1)+'.csv',index=False)

    if(tta+1==16):

        df_tmp = df.copy()

        df_tmp['label'] /= 16

        df_tmp.to_csv('submission_tta_'+str(tta+1)+'.csv',index=False)



df['label'] /= num_tta

df.to_csv('submission_tta_'+str(num_tta)+'.csv',index=False)