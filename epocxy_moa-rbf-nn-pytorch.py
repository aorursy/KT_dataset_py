# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns; sns.set(color_codes=True)

import torch

import torch.nn.functional as F

from torch import nn, optim

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, models

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from sklearn import model_selection

import albumentations

import pandas as pd

import numpy as np

import io,skimage

import skimage.transform

from torch.utils.data import Dataset, DataLoader

import os,cv2,time

import gc,collections

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision

import copy

from torch.autograd import Variable

from torchvision import datasets, models, transforms

from torch import topk

from sklearn.model_selection import KFold

gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
import warnings

warnings.filterwarnings('ignore')
import random

def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(seed=42)
df_train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

df_train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

df_test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



print('df_train_features Shape',df_train_features.shape)

print('df_train_targets_scored Shape',df_train_targets_scored.shape)

print('df_test_features Shape',df_test_features.shape)
df_train_features['cp_type'] = df_train_features['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

df_train_features['cp_dose'] = df_train_features['cp_dose'].map({'D1': 3, 'D2': 4})

df_train_features['cp_time'] = df_train_features['cp_time'].map({24: 0, 48: 1, 72: 2})



## Test

df_test_features['cp_type'] = df_test_features['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

df_test_features['cp_dose'] = df_test_features['cp_dose'].map({'D1': 3, 'D2': 4})

df_test_features['cp_time'] = df_test_features['cp_time'].map({24: 0, 48: 1, 72: 2})
df_train_targets_scored = df_train_targets_scored.loc[df_train_features['cp_type'] == 0].reset_index(drop=True)

df_train_features = df_train_features.loc[df_train_features['cp_type']==0].reset_index(drop=True)

df_train_features.shape,df_train_targets_scored.shape
col_obj = list()

col_float = list()

for col in df_train_features.columns[1:]:

    if df_train_features[col].dtypes == float:

        col_float.append(col)

    else:

        col_obj.append(col)

assert len(col_float) + len(col_obj) == df_train_features.shape[1] - 1
df_train = pd.merge(df_train_features,df_train_targets_scored,on='sig_id')

x_cols = df_train_features.columns[1:]

y_cols = df_train_targets_scored.columns[1:]



print('Features',len(x_cols))

print('Labels',len(y_cols))
df_train['folds'] = -1

df_train = df_train.sample(frac=1).reset_index(drop=True)

splits = 5

kf = KFold(n_splits=splits,shuffle = False)

for fold,(train_index, val_idx) in enumerate(kf.split(df_train)):

    #df.iloc[train_index,:]['kfold'] = int(fold+1)

    df_train.loc[val_idx,'folds'] = int(fold)



print('Number of Unique folds in dataset',df_train['folds'].unique())
class Train_Dataset(Dataset):

    def __init__(self, dataframe,features_col,labels ):

        

        self.dataframe = dataframe

        self.features_col = features_col

        self.labels = labels

        

                

        self.x = self.dataframe[self.features_col].values

        self.y = self.dataframe[self.labels].values

        

    def __len__(self):

        return self.dataframe.shape[0]

    

    def __getitem__(self, idx):

        

        feat = torch.FloatTensor(self.x[idx])

        lab = torch.FloatTensor(self.y[idx])

        

        return feat,lab

    

class Test_Dataset(Dataset):

    def __init__(self, dataframe,features_col):

        

        self.dataframe = dataframe

        self.features_col = features_col

                

                

        self.x = self.dataframe[self.features_col].values

        

        

    def __len__(self):

        return self.dataframe.shape[0]

    

    def __getitem__(self, idx):

        

        feat = torch.FloatTensor(self.x[idx])

        

        

        return feat
class RBF(nn.Module):





    def __init__(self, num_centers,in_features ,n_classes):

        super(RBF, self).__init__()

        

        self.num_centers = num_centers

        self.n_classes = n_classes

        self.in_features = in_features

        

        self.center = nn.Parameter(torch.Tensor(self.num_centers, self.in_features))

        self.center = nn.init.normal_(self.center, 0, 1)

        

        

        self.sigma = nn.Parameter(torch.Tensor(self.num_centers))

        self.sigma = nn.init.constant_(self.sigma, 1)

        

        self.linear =  nn.Sequential(nn.BatchNorm1d(self.num_centers)

                                ,nn.Dropout(.35)

                                ,nn.ReLU()

                                ,nn.Linear(self.num_centers, 1024)

                                ,nn.BatchNorm1d(1024)

                                ,nn.Dropout(.35)

                                ,nn.ReLU()

                                ,nn.Linear(1024 ,self.n_classes) )

        

        



        

    def function(self,inp):

        batch_size , num_centers , in_features= inp.size(0),self.num_centers,inp.size(1)

        x = inp.unsqueeze(1).expand(-1,num_centers,in_features)

        c = self.center.unsqueeze(0).expand(batch_size,num_centers,-1)

        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigma.unsqueeze(0)

        return distances

                

    def forward(self,inp):

        x = self.function(inp)

        x = self.linear(x)

        

        return x
num_centers = 2408

in_features = 875

n_classes = 206

model = RBF(num_centers,in_features ,n_classes).to(device)

model
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2)

loss_func = nn.BCEWithLogitsLoss()





pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of parameters',pytorch_total_params)
def train(loader,model):

    

    model.train()

    train_running_loss = 0

    for index,(feat,label) in enumerate(loader):

        x,y = feat.to(device),label.to(device)

        

        # Zero the parameter gradients

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_func(pred,y)

        loss.backward()

        optimizer.step()

        train_running_loss += loss.item()

        

    return train_running_loss/len(loader)





def valid(loader,model):

    

    

    valid_running_loss = 0

    pred_list = list()

    with torch.no_grad():

        model.eval()

        for val_index,(feat,label) in enumerate(loader):

            x,y = feat.to(device),label.to(device)

            

            # Zero the parameter gradients

            pred = model(x)

            pred_list.append(pred)

            loss = loss_func(pred,y)

            valid_running_loss += loss.item()

            

    return torch.cat(pred_list).sigmoid().detach().cpu().numpy(),valid_running_loss/len(loader)







def test(loader,model):

    labels = []

    with torch.no_grad():

        model.eval()

        for index,feat in enumerate(loader):

            x = feat.to(device)

            output = model(x)

            labels.append(output.sigmoid().detach().cpu().numpy())

    

    labels = np.concatenate(labels)        

    

    return labels
Batch_size = 64

n_epoch = 30

patience = 3

val_preds = np.zeros((df_train[y_cols].shape))



pred_labels = []

for fold_num in range(splits):

    best_score = None

    counter = 0

    print('='*30,'*****','Fold',fold_num+1,'*****','='*30)

    

    trn_idx = df_train[df_train['folds'] != fold_num].index

    val_idx = df_train[df_train['folds'] == fold_num].index

    

    df_trn = df_train.loc[trn_idx].reset_index(drop=True)

    df_val = df_train.loc[val_idx].reset_index(drop=True)

    

    ### Train Dataset

    train_dataset = Train_Dataset(df_trn,x_cols,y_cols)

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)

    

    ## Valid Dataset

    

    valid_dataset = Train_Dataset(df_val,x_cols,y_cols);

    valid_loader = DataLoader(valid_dataset, batch_size=Batch_size, shuffle=False)

    

        

    for epoch in range(n_epoch):

        print("Epoch: {}/{}.. ".format(epoch+1, n_epoch))

    

        train_loss = train(train_loader,model)

        v_out,valid_loss = valid(valid_loader,model)

        val_preds[val_idx] = v_out

    

        print(f'\tTrain Loss: {train_loss:.3f}')

        print(f'\t Val. Loss: {valid_loss:.3f}')

        

        if best_score is None:

            best_score = valid_loss

            print(f'if save model at epoch {epoch+1} ......')

        elif valid_loss > best_score:            

            best_score = valid_loss

            print(f'elif save model at epoch {epoch+1} ......')

        else:

            print('patience starts .........')

            counter +=1

            if counter > patience:

                print(f'else save model at epoch {epoch+1} ......')

                break;

        

    print(f'Inferenceing the test data at epoch {epoch+1} .....') 

    test_dataset = Test_Dataset(df_test_features,df_test_features.columns[1:])

    test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False)

    pred_labels.append(test(test_loader,model))
from sklearn.metrics import log_loss

score = 0

for i in range(df_train[y_cols].shape[1]):

    _score = log_loss(df_train[y_cols].iloc[:,i], val_preds[:,i])

    score += _score / df_train[y_cols].shape[1]

print(f"oof score: {score}")
sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

sub.iloc[:,1:] = 0 ## All labe;ls value to zero

test_labels = [sum(x)/len(x) for x in zip(*pred_labels)]

sub.iloc[:,1:] = test_labels

sub.head()
sub.to_csv('submission.csv', index=False)