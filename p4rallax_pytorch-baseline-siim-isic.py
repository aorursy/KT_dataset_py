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
seed = 2019

device = 'cuda:0'
data_path ='../input/siim-isic-melanoma-classification'
train = pd.read_csv(f'{data_path}/train.csv')

train.head()
imagelist = glob.glob(f'{data_path}/jpeg/train/*')

imagelist[0], len(imagelist)
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

        

        
def plotimgs(dataset):

    f , ax = plt.subplots(1,3)

    for p in range(3):

        idx = np.random.randint(0 , len(dataset))

        data = dataset[idx]

        img = data['image']

        ax[p].imshow(np.transpose(img,(1,2,0)), interpolation = 'nearest')

        ax[p].set_title(idx)
train_transforms = A.Compose([A.Flip(p=0.6),

                              A.ShiftScaleRotate(p=0.7),

                              A.Normalize(),

                              ToTensor()])

    

valid_transforms = A.Compose([A.Normalize(),

                             ToTensor()])
img_dir = '../input/siim-isic-melanoma-classification/jpeg/train'
def get_dataloader(df , img_dir ,mode, size, batch_size):

    dataset = SIIMDataset( df, img_dir ,mode ,transform = train_transforms if mode=='train' else valid_transforms , size=size)

    istrain = mode == 'train'

    dataloader = DataLoader(dataset , batch_size = batch_size , num_workers = 4 , shuffle = istrain ,drop_last = istrain)

    return dataloader
train_0 = train[train['target'] == 0]

train_1 = train[train['target']== 1]

train_0.shape , train_1.shape
train_0_new =train_0.sample(584)

train_0_new.shape
balanced = pd.concat([train_0_new ,train_1])

balanced['target'].value_counts()
train_df , valid_df = train_test_split(balanced,  test_size = 0.1 , stratify = balanced['target'].values ,random_state = seed)

train_df.shape , valid_df.shape
from torch.nn.parameter import Parameter



def gem(x, p=3, eps=1e-6):

    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)



class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):

        super(GeM,self).__init__()

        self.p = Parameter(torch.ones(1)*p)

        self.eps = eps

    def forward(self, x):

        return gem(x, p=self.p, eps=self.eps)       

    def __repr__(self):

        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
def GlobalAveragePooling(x):

    return x.mean(axis=-1).mean(axis=-1)
class CustomNet(nn.Module):

    def __init__(self , num_classes):

        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b3')

        #self.gem = GeM()

        self.pool= GlobalAveragePooling

        self.out = nn.Linear(1536, num_classes)

    

    def forward(self,x):

        x = self.model.extract_features(x)

        #x = F.avg_pool2d(x, x.size()[2:]).reshape(-1, 1536)

        x = self.pool(x)

        return self.out(x)
def  update_avg(curr_avg , val , idx):

    return (curr_avg * idx + val )/ (idx+1)
def train_epoch(model , dataloader , criterion , optimizer , scheduler ):

    model.train()

    curr_avg_loss = 0

    #train_losses = []

    #avg_losses = []

    t = tqdm.tqdm_notebook(dataloader , total = len(dataloader))

    for bi , data in enumerate(t):

        images = data['image'].cuda()

        labels = data['label'].cuda()

        scheduler.step()

        logits = model(images)

        loss = criterion(logits.squeeze() , labels.float())

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        curr_avg_loss = update_avg(curr_avg_loss , loss , bi)

        t.set_description('loss : %.5f , lr : %.6f' % (curr_avg_loss.item(), optimizer.param_groups[0]['lr']))

        

        return curr_avg_loss.item()

    
def validate(model , dataloader , criterion, ):

    model.eval()

    curr_avg_loss = 0

    #val_avg_losses = []

    #val_losses = []

    val_preds , val_targets = [] , []

    t =tqdm.tqdm_notebook(dataloader , total = len(dataloader))

    with torch.no_grad():

        for bi , data in enumerate(t):

            images = data['image'].cuda()

            labels = data['label'].cuda()

            logits = model(images)

            probs = torch.sigmoid(logits)

            val_preds.append(probs.detach().cpu().numpy())

            val_targets.append(labels.detach().cpu().numpy())

            loss = criterion(logits.squeeze() , labels.float())

            curr_avg_loss = update_avg(curr_avg_loss , loss , bi)

                    

            t.set_description('val loss: {:.4}'.format(curr_avg_loss.item()))

    val_preds = np.concatenate(val_preds)

    val_targets = np.concatenate(val_targets)

    metric = roc_auc_score(val_targets , val_preds)

    

    

    return curr_avg_loss.item() , metric

    

        

    
train_loader = get_dataloader(train_df , img_dir , mode='train' , size=256 , batch_size = 32)

valid_loader = get_dataloader(valid_df ,img_dir , mode = 'valid' , size=256 , batch_size = 64)
model = CustomNet(num_classes = 1)
model.to(device)
num_epochs = 30

lr = 1e-3

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=5000)
best_val_score = 0.0

es = 0 #early stopping

train_losses = []

val_losses =[]

metrics =[]



for epoch in range(num_epochs):

    train_loss = train_epoch(model, train_loader , criterion , optimizer ,scheduler )

    valid_loss , roc = validate(model ,valid_loader , criterion )

    train_losses.append(train_loss)

    val_losses.append(valid_loss)

    metrics.append(roc)

    print(f'Epoch: {epoch+1} | Train_loss: {train_loss:.5f} | Val loss: {valid_loss:.5f} | roc_metric : {roc:.5f}')

    

    if roc > best_val_score:

        es = 0

        best_val_score = roc

        torch.save(model.state_dict(), 'model.pth')

        

    else:

        es += 1

        if es == 10:

            break
col_names = [ 'Train Loss' , 'Val Loss'  , 'Val ROC-AUC']

stats = pd.DataFrame(np.stack([train_losses  ,val_losses  ,metrics], axis =1), columns = col_names)
stats