!pip install -q efficientnet_pytorch
import os

import cv2

import glob

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import tqdm

import torch

import torch.nn.functional as F

import torch.nn as nn

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

import albumentations as A

from albumentations.pytorch import ToTensor

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from transformers import get_cosine_schedule_with_warmup

from efficientnet_pytorch import EfficientNet
class SIIMDataset(Dataset):

    def __init__(self, df, data_path , mode= 'train', transform = None , size=256):

        self.df = df

        self.image_ids = df['image_name'].tolist()

        self.data_path = data_path

        self.mode= mode

        self.transform = transform

        self.size = size

        

    def __len__(self):

        return len(self.image_ids)

    

    def __getitem__(self , idx):

        image_id = self.image_ids[idx]

        image_path = os.path.join(self.data_path , image_id + '.jpg')

        image = cv2.imread(image_path)

        image = cv2.resize(image, (self.size,self.size))

        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

        

        if self.transform:

            aug = self.transform(image=image)

            image= aug['image']

            

        data = {}

        data['image'] = image

        data['image_id'] = image_id

        

        

        if self.mode == 'test':

            return data

        else:

            label = self.df.loc[self.df['image_name'] == image_id , 'target'].values[0]

            data['label'] = torch.tensor(label)

            return data
valid_transforms = A.Compose([A.Normalize(),

                             ToTensor()])
def plotimgs(dataset):

    f , ax = plt.subplots(1,3)

    for p in range(3):

        idx = np.random.randint(0 , len(dataset))

        data = dataset[idx]

        img = data['image']

        ax[p].imshow(np.transpose(img,(1,2,0)), interpolation = 'nearest')

        ax[p].set_title(idx)
img_path = '../input/siim-isic-melanoma-classification/'

test_path = '../input/siim-isic-melanoma-classification/jpeg/test'

device = 'cuda:0'
test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
testdata = SIIMDataset(test_df ,test_path, mode='test' , transform = valid_transforms , size = 256)
test_loader = DataLoader(testdata , batch_size = 64 , shuffle = False )
class CustomNet(nn.Module):

    def __init__(self , num_classes):

        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b3')

        #self.gem = GeM()

        self.out = nn.Linear(1536, num_classes)

    

    def forward(self,x):

        x = self.model.extract_features(x)

        x = F.avg_pool2d(x, x.size()[2:]).reshape(-1, 1536)

        #x = gem(x)

        return self.out(x)
model = CustomNet(num_classes = 1)

model.load_state_dict(torch.load('../input/pytorch-baseline-siim-isic/model.pth'))

model.to(device)
model.eval()

preds = []

t= tqdm.tqdm_notebook(test_loader, total=len(test_loader))

with torch.no_grad():

    for bi,data in enumerate(t):

        images = data['image'].cuda()

        preds.extend(model(images).squeeze().detach().cpu().numpy())
sample = pd.read_csv(img_path + 'sample_submission.csv')
def sigmoid(x):

    return 1/(1 + np.exp(-x))
sample.target = sigmoid(np.array(preds))
sample.to_csv('submission.csv', index = False)