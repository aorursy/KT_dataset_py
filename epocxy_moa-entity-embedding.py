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

import pandas as pd

import numpy as np

from torch.utils.data import Dataset, DataLoader

import gc,collections,os,cv2,time,copy

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision

from torch.autograd import Variable

from torchvision import datasets, models, transforms

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
print('df_train_features columns',len(df_train_features.columns))

print('df_train_features columns',len(df_train_targets_scored.columns))
## Train

df_train_features['cp_type'] = df_train_features['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

#df_train_features['cp_dose'] = df_train_features['cp_dose'].map({'D1': 3, 'D2': 4})

#df_train_features['cp_time'] = df_train_features['cp_time'].map({24: 0, 48: 1, 72: 2})



## Test

df_test_features['cp_type'] = df_test_features['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

#df_test_features['cp_dose'] = df_test_features['cp_dose'].map({'D1': 3, 'D2': 4})

#df_test_features['cp_time'] = df_test_features['cp_time'].map({24: 0, 48: 1, 72: 2})
df_train_targets_scored = df_train_targets_scored.loc[df_train_features['cp_type'] == 0].reset_index(drop=True)

df_train_features = df_train_features.loc[df_train_features['cp_type']==0].reset_index(drop=True)

df_train_features.shape,df_train_targets_scored.shape
df_test_features['cp_type'].value_counts()
df_train_features['cp_time_dose'] =  df_train_features['cp_time'].map(str) + df_train_features['cp_dose'].map(str)

df_test_features['cp_time_dose'] =  df_test_features['cp_time'].map(str) + df_test_features['cp_dose'].map(str)
## droping columns

df_train_features = df_train_features.drop(['cp_type' ,'cp_time', 'cp_dose'], axis=1)

df_test_features = df_test_features.drop(['cp_type','cp_time', 'cp_dose'], axis=1)

print('df_train_features Shape',df_train_features.shape)

print('df_test_features Shape',df_test_features.shape)
df_train = pd.merge(df_train_features,df_train_targets_scored,on='sig_id')

x_cols = df_train_features.columns[1:]

y_cols = df_train_targets_scored.columns[1:]



print('Features',len(x_cols))

print('Labels',len(y_cols))
### For K-fold Validation

df_train['folds'] = -1

df_train = df_train.sample(frac=1).reset_index(drop=True)

splits = 5

kf = KFold(n_splits=splits,shuffle = False)

for fold,(train_index, val_idx) in enumerate(kf.split(df_train)):

    #df.iloc[train_index,:]['kfold'] = int(fold+1)

    df_train.loc[val_idx,'folds'] = int(fold)



print('Number of Unique folds in dataset',df_train['folds'].unique())
cat_cols = [col for col in x_cols if col is 'cp_time_dose']

num_cols = [col for col in x_cols if col is not  'cp_time_dose']

print('Length of cat_cols',len(cat_cols))

print('Length of num_cols',len(num_cols))
assert df_train['cp_time_dose'].nunique() == df_test_features['cp_time_dose'].nunique()



train_cate_list  = []

test_cate_list   = []

raw_vals = np.unique(df_train[cat_cols])

val_map = {}

for i in range(len(raw_vals)):

    val_map[raw_vals[i]] = i



train_cate_list = df_train['cp_time_dose'].map(val_map).values

test_cate_list = df_test_features['cp_time_dose'].map(val_map).values
class Train_Dataset(Dataset):

    def __init__(self, df, cont_feat, cat_list , labels ):

        

        self.df = df

        self.cont_values = self.df[cont_feat].values        

        self.labels      = self.df[labels].values

        self.train_cate_list = cat_list

        



          

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self,idx):

        

        numeric_feat = torch.FloatTensor(self.cont_values[idx])

        category_feat_list = self.train_cate_list[idx]

        

        lab = torch.Tensor(self.labels[idx])

        '''

        dict_ = {

            'cont' : numeric_feat,

            'cat' : category_feat,

            'label': lab

        }

        '''

        return numeric_feat,category_feat_list,lab

    

class Test_Dataset(Dataset):

    def __init__(self,  df, cont_feat, cat_list):

        

        self.df = df

        self.cont_values = self.df[cont_feat].values        

        self.train_cate_list = cat_list

        

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        

        numeric_feat = torch.FloatTensor(self.cont_values[idx])

        category_feat_list = self.train_cate_list[idx]

        

        return numeric_feat , category_feat_list

        

        
class embed_NN(nn.Module):

    def __init__(self, col_shape , embed_dim , source_input_dim , hidden_unit , target_dim):

        super().__init__()

        

        self.embed = nn.Embedding(col_shape ,embed_dim)

        

        self.dropout = nn.Dropout(.35)

        

        self.fc1 = nn.Sequential(nn.Linear(source_input_dim , hidden_unit)

                                ,nn.BatchNorm1d(hidden_unit)

                                ,nn.Dropout(.35)

                                ,nn.PReLU()

                                )

        self.fc2 = nn.Sequential((nn.Linear(hidden_unit + embed_dim , hidden_unit//3))

                                ,nn.BatchNorm1d(hidden_unit//3)

                                ,nn.Dropout(.35)

                                ,nn.PReLU() )                  

                                

        self.fc3 = nn.Linear(hidden_unit//3 , target_dim)             

                                

    def forward(self , xcat , xcont):

        cate =  self.dropout(self.embed(xcat))

        conti = self.fc1(xcont)

        concat = torch.cat([conti , cate],1)

        x = self.fc2(concat)

        x = self.fc3(x)

        return x
col_shape = df_train.shape[0]

embed_dim = 4

source_input_dim = len(num_cols)

hidden_unit = 512

target_dim =len(y_cols)



model = embed_NN(col_shape , embed_dim ,source_input_dim, hidden_unit , target_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=3, verbose=True, factor=0.2)

loss_func = nn.BCEWithLogitsLoss()





pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of parameters',pytorch_total_params)
def train(loader,model):

    

    model.train()

    train_running_loss = 0

    for index,(cont,cat,labels) in enumerate(train_loader):

        xcont,xcat,ylabels = cont.to(device) , cat.to(device) , labels.to(device)



        # Zero the parameter gradients

        optimizer.zero_grad()

        pred = model(xcat , xcont)

        loss = loss_func(pred,ylabels)

        loss.backward()

        optimizer.step()

        train_running_loss += loss.item()

        

    return train_running_loss/len(loader)





def valid(loader,model):

    

    

    valid_running_loss = 0

    pred_list = list()

    with torch.no_grad():

        model.eval()

        for val_index,(cont,cat,labels) in enumerate(loader):

            xcont,xcat,ylabels = cont.to(device) , cat.to(device) , labels.to(device)

            

            # Zero the parameter gradients

            pred = model(xcat , xcont)

            pred_list.append(pred)

            loss = loss_func(pred,ylabels)

            valid_running_loss += loss.item()

            

    return torch.cat(pred_list).sigmoid().detach().cpu().numpy(),valid_running_loss/len(loader)







def test(loader,model):

    labels = []

    with torch.no_grad():

        model.eval()

        for tst_index,(cont,cat) in enumerate(loader):

            xcont,xcat = cont.to(device) , cat.to(device)

            output = model(xcat , xcont)

            labels.append(output.sigmoid().detach().cpu().numpy())

    

    labels = np.concatenate(labels)        

    

    return labels
Batch_size = 64

n_epoch = 30

patience = 5

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

    train_dataset = Train_Dataset(df_trn , num_cols, train_cate_list , y_cols )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    

    ## Valid Dataset

    

    valid_dataset = Train_Dataset(df_val , num_cols, train_cate_list , y_cols );

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

    test_dataset = Test_Dataset(df_test_features, num_cols, train_cate_list)

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