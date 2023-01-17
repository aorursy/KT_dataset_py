!pip install pretrainedmodels
!pip install albumentations
# for TPU
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
import os
import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pretrainedmodels

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from albumentations import Rotate 

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import warnings  
warnings.filterwarnings('ignore')
# for TPU
import torch_xla
import torch_xla.core.xla_model as xm
DIR_INPUT = '/kaggle/input/plant-pathology-2020-fgvc7'

SEED = 27

# If you want to chane parameters, check this for settings.
N_FOLDS = 5
N_EPOCHS = 10
BATCH_SIZE = 64
SIZE = 512
# for GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for TPU
device = xm.xla_device()
torch.set_default_tensor_type('torch.FloatTensor')
class PlantDataset(Dataset):
    
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms=transforms
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        image_src = DIR_INPUT + '/images/' + self.df.loc[idx, 'image_id'] + '.jpg'
        # print(image_src)
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        labels = torch.from_numpy(labels.astype(np.int8))
        labels = labels.unsqueeze(-1)
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, labels
transforms_train = A.Compose([
        A.RandomResizedCrop(height=SIZE, width=SIZE, p=1.0),
        A.Rotate(20),
        A.Flip(),
        A.Transpose(),
    A.Resize(height=SIZE, width=SIZE, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
    ], p=1.0)

transforms_valid = A.Compose([
    A.Resize(height=SIZE, width=SIZE, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])
train_df = pd.read_csv(DIR_INPUT + '/train.csv')
train_labels = train_df.iloc[:, 1:].values

# Need for the StratifiedKFold split
train_y = train_labels[:, 2] + train_labels[:, 3] * 2 + train_labels[:, 1] * 3
train_df.head()
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros((train_df.shape[0], 4))
model_name = 'resnet34'
model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
in_features = model.last_linear.in_features
model.last_linear = nn.Linear(in_features, 4)
class DenseCrossEntropy(nn.Module):

    def __init__(self):
        super(DenseCrossEntropy, self).__init__()
        
        
    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()
        
        logprobs = F.log_softmax(logits, dim=-1)
        
        loss = -labels * logprobs
        loss = loss.sum(-1)

        return loss.mean()
def train_one_fold(i_fold, model, criterion, optimizer, lr_scheduler, dataloader_train, dataloader_valid):
    
    train_fold_results = []

    for epoch in range(N_EPOCHS):

        print('  Epoch {}/{}'.format(epoch + 1, N_EPOCHS))
        print('  ' + ('-' * 20))
        os.system(f'echo \"  Epoch {epoch}\"')

        model.train()
        tr_loss = 0

        for step, batch in enumerate(dataloader_train):

            images = batch[0].to(device, dtype=torch.float)
            labels = batch[1].to(device, dtype=torch.float)
            
            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(-1))                
            loss.backward()

            tr_loss += loss.item()
            
            # for TPU
            #optimizer.step()
            xm.optimizer_step(optimizer, barrier=True)
            
            optimizer.zero_grad()

        # Validate
        model.eval()
        
        val_loss = 0
        val_preds = None
        val_labels = None

        for step, batch in enumerate(dataloader_valid):

            images = batch[0].to(device, dtype=torch.float)
            labels = batch[1].to(device, dtype=torch.float)

            if val_labels is None:
                val_labels = labels.clone().squeeze(-1)
            else:
                val_labels = torch.cat((val_labels, labels.squeeze(-1)), dim=0)

            

            with torch.no_grad():
                outputs = model(batch[0].to(device))

                loss = criterion(outputs, labels.squeeze(-1))
                val_loss += loss.item()

                preds = torch.softmax(outputs, dim=1).data.cpu()
                #preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()

                
                if val_preds is None:
                    val_preds = preds
                else:
                    val_preds = torch.cat((val_preds, preds), dim=0)
           
        # if train mode
        lr_scheduler.step(tr_loss)

        train_fold_results.append({
            'fold': i_fold,
            'epoch': epoch,
            'train_loss': tr_loss / len(dataloader_train),
            'valid_loss': val_loss / len(dataloader_valid),
            'valid_score': roc_auc_score(val_labels.view(-1).cpu(), val_preds.view(-1).cpu(), average='macro'),
        })

    return val_preds, train_fold_results
submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')
submission_df.iloc[:, 1:] = 0
dataset_test = PlantDataset(df=submission_df, transforms=transforms_valid)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
submissions = None
train_results = []

device = xm.xla_device()

for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_y)):
    print("Fold {}/{}".format(i_fold + 1, N_FOLDS))

    valid = train_df.iloc[valid_idx]
    valid.reset_index(drop=True, inplace=True)

    train = train_df.iloc[train_idx]
    train.reset_index(drop=True, inplace=True)    

    dataset_train = PlantDataset(df=train, transforms=transforms_train)
    dataset_valid = PlantDataset(df=valid, transforms=transforms_valid)

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    # device = torch.device("cuda:0")    
    model = model.to(device)

    criterion = DenseCrossEntropy()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(N_EPOCHS * 0.5), int(N_EPOCHS * 0.75)], gamma=0.1, last_epoch=-1)
    
    val_preds, train_fold_results = train_one_fold(i_fold, model, criterion, optimizer, lr_scheduler, dataloader_train, dataloader_valid)
    oof_preds[valid_idx, :] = val_preds
    
    train_results = train_results + train_fold_results

    model.eval()
    test_preds = None

    for step, batch in enumerate(dataloader_test):

        images = batch[0].to(device, dtype=torch.float)

        with torch.no_grad():
            outputs = model(images)

            if test_preds is None:
                test_preds = outputs.data.cpu()
            else:
                test_preds = torch.cat((test_preds, outputs.data.cpu()), dim=0)
    
    
    # Save predictions per fold
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_preds, dim=1)
    submission_df.to_csv('submission_fold_{}.csv'.format(i_fold), index=False)

    # logits avg
    if submissions is None:
        submissions = test_preds / N_FOLDS
    else:
        submissions += test_preds / N_FOLDS

print("5-Folds CV score: {:.4f}".format(roc_auc_score(train_labels, oof_preds, average='macro')))

torch.save(model.state_dict(), '5-folds_rnn34.pth')
import matplotlib.pyplot as plt
def show_training_loss(train_result):
    plt.figure(figsize=(15,10))
    plt.subplot(3,1,1)
    train_loss = train_result['train_loss']
    plt.plot(train_loss.index, train_loss, label = 'train_loss')
    plt.legend()

    val_loss = train_result['valid_loss']
    plt.plot(val_loss.index, val_loss, label = 'val_loss')
    plt.legend()
def show_valid_score(train_result):
    plt.figure(figsize=(15,10))
    plt.subplot(3,1,1)
    valid_score = train_result['valid_score']
    plt.plot(valid_score.index, valid_score, label = 'valid_score')
    plt.legend()
train_results = pd.DataFrame(train_results)
train_results.head(10)
submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(submissions, dim=1)
submission_df.to_csv('submission.csv', index=False)