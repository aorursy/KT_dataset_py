!unzip -qq /kaggle/input/pandatiles/train
!pip install /kaggle/input/pretrainedmodelswhl/pretrainedmodels-0.7.4-py3-none-any.whl
!pip install iterative-stratification
import os

import numpy as np
import pandas as pd

import openslide
import torch
import cv2
from tqdm.auto import tqdm
import albumentations
import pretrainedmodels

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.utils.data as data_utils
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from fastai.layers import LabelSmoothingCrossEntropy

from sklearn.metrics import cohen_kappa_score
# Some constants
BASE_DIR = '/kaggle/input/prostate-cancer-grade-assessment'
DATA_DIR = '/kaggle/working'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.01
SEED = 28
FOLDS = 5
N_IMAGES = 16
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
train_df.head()
train_df.loc[:, 'kfold'] = -1

X = train_df[['image_id', 'data_provider']].values
y = train_df[['isup_grade', 'gleason_score']].values

mskf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
    train_df.loc[val_idx, 'kfold'] = fold

print(train_df.kfold.value_counts())
train_df.head()
train_df['gleason_score'].unique()
train_df['gleason_score'] = train_df['gleason_score'].str.replace('negative', '0+0')
train_df['gleason_score'].unique()
train_df['gleason_score'] = train_df['gleason_score'].astype('category')
train_df.dtypes
mappings = dict(enumerate(train_df['gleason_score'].cat.categories))
mappings
train_df['gleason_score'] = train_df['gleason_score'].cat.codes
train_df.dtypes
print(f'Number of classes: {len(train_df["gleason_score"].unique())}')
class PandaDataset(Dataset):
    """Custom dataset for PANDA"""
    
    def __init__(self, df, folds, mean=(1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304), std=(0.36357649, 0.49984502, 0.40477625)):
        self.df = df
        self.df = self.df[self.df.kfold.isin(folds)].reset_index(drop=True)
        
        # In case of validation dataset, don't apply transformations
        if len(folds) == 1:
            self.aug = albumentations.Compose([
                albumentations.Normalize(mean, std, always_apply=True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Normalize(mean, std, always_apply=True)
            ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_id = self.df.loc[index]['image_id']
        
        # Read all 16 images and create a large image
        arr = [[], [], [], []]
        row = 0
        for i in range(N_IMAGES):
            image = cv2.imread(os.path.join(DATA_DIR, f'{image_id}.tiff_{i}.png'))
            if i % 4 == 0:
                row += 1
            arr[row-1].append(image)

        for i in range(len(arr)):
            arr[i] = np.hstack(arr[i])

        full_image = np.vstack(arr)
        full_image = self.aug(image=full_image)['image']
        
        # Convert from NHWC to NCHW as pytorch expects images in NCHW format
        full_image = np.transpose(full_image, (2, 0, 1))
        
        # For now, just return image and ISUP grades
        return full_image, self.df.loc[index]['gleason_score']
# temp_train_df = train_df.iloc[0:10]
# temp_train_df
train_dataset = PandaDataset(train_df, folds=[1, 2, 3, 4])
train_loader = data_utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = PandaDataset(train_df, folds=[0])
val_loader = data_utils.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
class SEResNext50(nn.Module):
    """
    Define SEResNext50 model with 10 output classes based on gleason scores
    """
    def __init__(self, pretrained):
        super(SEResNext50, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=None)
        
        self.l0 = nn.Linear(2048, 10)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        return l0
model = SEResNext50(pretrained=True)
model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-08)
criterion = LabelSmoothingCrossEntropy(0.1)  # Using label smoothing loss instead of normal cross entropy loss
val_acc_list, qwks = [], []

for epoch in range(EPOCHS):
    model.train()
    for i, (x_train, y_train) in tqdm(enumerate(train_loader), total=int(len(train_dataset)/train_loader.batch_size)):
        x_train = x_train.to(DEVICE, dtype=torch.float32)/255
        y_train = y_train.to(DEVICE, dtype=torch.long)
        
        
        # Forward pass
        preds = model(x_train)
        loss = criterion(preds, y_train)
        
        # Backpropagate
        optimizer.zero_grad()  # Reason: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        loss.backward()
        optimizer.step()
    
    lr_scheduler.step(loss.item())

    # Calculate validation accuracy after each epoch
    # Predict on validation set
    
    with torch.no_grad():
        model.eval()

        correct = 0
        val_size = len(val_loader)
        all_targets, all_preds = [], []
        for x_val, y_val in tqdm(val_loader, total=int(len(val_dataset)/val_loader.batch_size)):

            x_val = x_val.to(DEVICE, dtype=torch.float32)/255
            y_val = y_val.to(DEVICE, dtype=torch.long)

            val_preds = model(x_val)
            _, preds = torch.max(val_preds.data, axis=1)
            correct += (preds == y_val).sum().item()
            
            all_targets.append(y_val.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

        val_acc = correct / len(val_dataset)
        val_acc_list.append(val_acc)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        qwk = cohen_kappa_score(all_preds, all_targets, weights='quadratic')
        qwks.append(qwk)

    print('Epoch [{}/{}], Loss: {:.4f}, QWK: {} Validation accuracy: {:.2f}%'
          .format(epoch + 1, EPOCHS, loss.item(), qwk, val_acc * 100))
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, len(val_acc_list)), val_acc_list, label='Validation Accuracy')

plt.title('Accuracy')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
print(qwks)
# Remove all the images from working directory to prevent them from kernel's output
!rm -rf *
torch.save(model.state_dict(), 'seresnext_ls_fold0.pth')