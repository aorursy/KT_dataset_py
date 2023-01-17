import numpy as np
import pandas as pd
import json
%%time
# prints cell time

# Train data
with open('../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json', "r", encoding="ISO-8859-1") as file:
    train = json.load(file)

# remove image_id, license and region_id columns because they are unnecessary
train_img = pd.DataFrame(train['images']).drop(columns='license')
train_ann = pd.DataFrame(train['annotations']).drop(columns=['image_id', 'region_id'])
# final data frame
train_df = train_img.merge(train_ann, on='id')
train_df.head()
train_df['category_id'].value_counts()
CATEGORY_CLASSES = 32093

# set all labels by category_id
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(train_df['category_id'])
train_df['category_id_le'] = le.transform(train_df['category_id'])
class_map = dict(sorted(train_df[['category_id_le', 'category_id']].values.tolist()))
%%time

# Test data
with open('../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json', "r", encoding="ISO-8859-1") as file:
    test = json.load(file)

test_df = pd.DataFrame(test['images']).drop(columns='license')
test_df.head()
import os
import random
import gc
gc.enable()
import time

# image loading
import cv2
from PIL import Image

# f1 score
from sklearn.metrics import f1_score

# progress bar
from tqdm import tqdm

# main library with neraul network for our image processing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

# image transformations
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

# set PyTorch processing with gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set seed for same result
SEED = 999

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# necessary option for same behaviour
torch.backends.cudnn.deterministic = True
# image transformation function
HEIGHT = 200
WIDTH = 200

def get_transforms():
    
    # Compose - use multiple transformations
    return Compose([
            Resize(HEIGHT, WIDTH),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

# Train set class
class TrainDataset(Dataset):
    # Train data frame, category labels, transform function
    def __init__(self, df, labels, transform=None):
        self.df = df
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # load image and convert it to neccessary for 'albumentations' format
        file_name = self.df['file_name'].values[idx]
        file_path = f'../input/herbarium-2020-fgvc7/nybg2020/train/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels.values[idx]
        
        # transform and normalize image
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
# create 0 and 1 folds to then split for Train and Validation sets
from sklearn.model_selection import StratifiedKFold

# folds = train_df.sample(n=200000, random_state=0).reset_index(drop=True).copy()
folds = train_df.copy()
train_labels = folds['category_id'].values
kf = StratifiedKFold(n_splits=2)
for fold, (train_index, val_index) in enumerate(kf.split(folds.values, train_labels)):
    folds.loc[val_index, 'fold'] = int(fold)
folds['fold'] = folds['fold'].astype(int)
folds.to_csv('folds.csv', index=None)
folds.head()
folds.shape
FOLD = 0
# Train and Validation indices
trn_idx = folds[folds['fold'] != FOLD].index
val_idx = folds[folds['fold'] == FOLD].index

# Train and Validation data sets
train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True), 
                             folds.loc[trn_idx]['category_id'], 
                             transform=get_transforms())
valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True), 
                             folds.loc[val_idx]['category_id'], 
                             transform=get_transforms())
batch_size = 512

# decorators for data sets for easy iteration (must implement __len__ and __getitem__)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# image recognition with Deep Residual Learning
model = models.resnet18(pretrained=True)
# max pooling
model.avgpool = nn.AdaptiveAvgPool2d(1)
# 2 layers classifier
model.fc = nn.Linear(model.fc.in_features, CATEGORY_CLASSES)
# Train proccess
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# number of epochs
n_epochs = 1
# learning rate
lr = 4e-4

model.to(device)

# stochastic optimization
optimizer = Adam(model.parameters(), lr=lr, amsgrad=False)
# optimize learning rate
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=5, verbose=True, eps=1e-6)

# cross entroy loss criterion
criterion = nn.CrossEntropyLoss()

best_score = 0.
best_loss = np.inf

for epoch in range(n_epochs):

    start_time = time.time()

    model.train()
    avg_loss = 0.
    
    # set gradients of model parameters to zero
    optimizer.zero_grad()
    
    # training
    for i, (images, labels) in tqdm(enumerate(train_loader)):

        # load to device
        images = images.to(device)
        labels = labels.to(device)
        
        # compute output
        y_preds = model(images)
        loss = criterion(y_preds, labels)
        
        # backward propogation
        loss.backward()
        # adjust weights
        optimizer.step()
        optimizer.zero_grad()

        avg_loss += loss.item() / len(train_loader)
        
    # enable prediction mode
    model.eval()
    avg_val_loss = 0.
    preds = np.zeros((len(valid_dataset)))
    
    # validating
    for i, (images, labels) in enumerate(valid_loader):

        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            y_preds = model(images)

        preds[i * batch_size: (i+1) * batch_size] = y_preds.argmax(1).to('cpu').numpy()

        loss = criterion(y_preds, labels)
        avg_val_loss += loss.item() / len(valid_loader)
    
    # optimize learning rate
    scheduler.step(avg_val_loss)
    
    # epoch score
    score = f1_score(folds.loc[val_idx]['category_id'].values, preds, average='macro')

    elapsed = time.time() - start_time

    print(f'Epoch {epoch+1} avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  F1: {score:.6f}  time: {elapsed:.0f}s')

    if score>best_score:
        best_score = score
        print(f'Epoch {epoch+1} save best score: {best_score:.6f} Model')
        torch.save(model.state_dict(), f'fold_{FOLD}_best_score.pth')

    if avg_val_loss<best_loss:
        best_loss = avg_val_loss
        print(f'Epoch {epoch+1} save best loss: {best_loss:.4f} Model')
        torch.save(model.state_dict(), f'fold_{FOLD}_best_loss.pth')
# Test set class
class TestDataset(Dataset):
    # Test data frame and transformation function
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # load image and convert it to albumentations format
        file_name = self.df['file_name'].values[idx]
        file_path = f'../input/herbarium-2020-fgvc7/nybg2020/test/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # transform and normalize image
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image
# Test data set creation with specified batch size
BATCH_SIZE = 1024

test_dataset = TestDataset(test_df, transform=get_transforms())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# the same model
model = models.resnet18(pretrained=False)
model.avgpool = nn.AdaptiveAvgPool2d(1)
model.fc = nn.Linear(model.fc.in_features, CATEGORY_CLASSES)

# Trained weights path
weights_path = '../input/herbarium-2020-pytorch-resnet18-train/fold0_best_score.pth'
model.load_state_dict(torch.load(weights_path))
# testing
model.to(device)

preds = np.zeros((len(test_dataset)))
for i, images in tqdm(enumerate(test_loader)):
    images = images.to(device)
    
    with torch.no_grad():
        y_preds = model(images)
        
    preds[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = y_preds.argmax(1).to('cpu').numpy()
# take example submission
sample_submission = pd.read_csv('../input/herbarium-2020-fgvc7/sample_submission.csv')
# rewrite with own predictions
test_df['preds'] = preds.astype(int)
submission = sample_submission.merge(test_df.rename(columns={'id': 'Id'})[['Id', 'preds']], on='Id').drop(columns='Predicted')
submission['Predicted'] = submission['preds'].map(class_map)
submission = submission.drop(columns='preds')
submission.to_csv('submission.csv', index=False)
submission.head()