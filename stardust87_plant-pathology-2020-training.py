!pip install timm
import os, time

import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim

import timm

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt


import warnings  
warnings.filterwarnings('ignore')


DIR_INPUT = '/kaggle/input/plant-pathology-2020-fgvc7'
IMAGE_INPUT = '/kaggle/input/plant-pathology-2020-resized-images'

SEED = 42
N_FOLDS = 5
N_EPOCHS = 20
BATCH_SIZE = 8
IMAGE_SIZE = (409,273)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
device
class PlantDataset(Dataset):
    
    def __init__(self, df, transforms=None):
    
        self.df = df
        self.transforms = transforms
        if not self.transforms:
            self.transforms  = A.Compose([ToTensorV2(p=1.0)])
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        image_src = IMAGE_INPUT + '/images_409_273/' + self.df.loc[idx, 'image_id'] + '.jpg'
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = self.transforms(image=image)
        image = transformed['image']
            
        labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        labels = torch.from_numpy(labels.astype(np.int8))
        labels = labels.unsqueeze(-1)
        
        

        return image, labels
def trim_network_at_index(network,index=-1):
    assert index <0, f'Param index must be negative. Received {index}'
    return nn.Sequential(*list(network.children())[:index])
class PlantModel(nn.Module):
    
    def __init__(self, num_classes=4):
        super().__init__()
        
#       resnets models (resnet50, resnest269e)
        self.backbone = timm.create_model('resnest269e',pretrained=True)
#         self.backbone = torchvision.models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
    
    
#         efficient model (efficientnet_b7_ns)
#         self.backbone = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)
#         in_features = self.backbone.classifier.in_features


        self.backbone = trim_network_at_index(self.backbone,-1) 
        self.logit = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)

        x = self.logit(x)

        return  x
train_df = pd.read_csv(DIR_INPUT + '/train.csv')
test_df = pd.read_csv(DIR_INPUT + '/test.csv')

train_labels = train_df.iloc[:, 1:].values
train_y = train_labels[:, 2] + train_labels[:, 3] * 2 + train_labels[:, 1] * 3

skf = StratifiedKFold(n_splits = N_FOLDS)
def show_image(sample):
    plt.imshow(sample[0].numpy().transpose(1, 2, 0))
transform = A.Compose([
    A.RandomBrightness(),
    A.Flip(),
    A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),

    # Pixels
    A.OneOf([
        A.IAAEmboss(p=1.0),
        A.IAASharpen(p=1.0),
        A.Blur(p=1.0),
    ], p=0.5),

    # Affine
    A.OneOf([
        A.ElasticTransform(p=1.0),
        A.IAAPiecewiseAffine(p=1.0)
    ], p=0.5),

    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

transforms_valid = A.Compose([
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0)
])
def train_fold(i_fold, trainloader,model,device,optimizer, criterion, scheduler):
    training_steps = len(trainloader)*N_EPOCHS
    progress_bar = tqdm(range(training_steps),ncols = '80%')
    loss_history = []
    train_iter = iter(trainloader)
    train_fold_results = []
    
    #Training
    for i in progress_bar:
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            data = next(train_iter)

        model.train()
        torch.set_grad_enabled(True)
        images,labels = data
        labels =  labels.squeeze().to(device)
        images = images.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs,labels.argmax(dim=1).long())
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        progress_bar.set_description(f'TRAINING Epoch: {i//(training_steps//N_EPOCHS)+1}/{N_EPOCHS}  Loss: {loss.item():.3f}  Avg_loss: {np.mean(loss_history[-10:]):.3f}')

     
    PATH = f'./modelF{i_fold}.pth'

    torch.save(model.state_dict(), PATH)
    
    train_fold_results.append({
            'fold': i_fold,
            'epoch': i//(training_steps//N_EPOCHS)+1,
            'train_loss': np.mean(loss_history[-30:]),
        })
    
    
    return train_fold_results
    
def validate_fold(i_fold, valloader,model,device, criterion):
    
    #Validation    
    model.eval()
    val_loss = []
    val_probs = []
    val_labels = []
    val_fold_results = []
    progress_bar = tqdm(range(len(valloader)),ncols = '80%')
    
    val_iter = iter(valloader)

    with torch.no_grad():
        for i in progress_bar:        
            data = next(val_iter)
            probs = F.softmax(model(data[0].to(device, dtype=torch.float)))
            labels = data[1].squeeze().to(device)
            
            loss = criterion(probs, labels.argmax(dim=1).long())
            val_loss.append(loss.item())
            
            val_probs.append(probs.cpu().numpy())
            val_labels.append(labels.cpu().numpy())
            
            progress_bar.set_description(f'VALIDATION   Loss: {loss.item():.3f}  Avg_loss: {np.mean(val_loss[-10:]):.3f}')
        
        
    val_labels = np.concatenate(val_labels, axis=0)
    val_probs = np.concatenate(val_probs, axis=0)
        

    val_fold_results.append({
    'fold': i_fold,
    'valid_loss': np.mean(val_loss),
    'valid_score': roc_auc_score(val_labels, val_probs, average='macro'),
    })

    
    return val_fold_results
train_results = []
val_results = []

start = time.perf_counter()
for i_fold, (train_index, val_index) in enumerate(skf.split(train_df,train_y)):

    train, val = train_df.loc[train_index], train_df.loc[val_index]
    val = val.reset_index(drop=True)
    train = train.reset_index(drop=True)
    
    dataset_train = PlantDataset(df=train,transforms = transform)
    dataset_val = PlantDataset(df=val, transforms = transforms_valid)
    
    trainloader = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=4,drop_last=True)
    valloader = DataLoader(dataset_val, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=4,drop_last=True)
    
    model = PlantModel()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, steps_per_epoch=len(trainloader), epochs=N_EPOCHS)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = 3e-6, max_lr=3e-4, step_size_up=4*len(trainloader),cycle_momentum = False, mode = 'exp_range')

    
    #Training
    train_fold_results = train_fold(i_fold, trainloader,model,device,optimizer, criterion, scheduler)
    train_results.append(train_fold_results)
    
               
        
    #Validation    
    val_fold_results = validate_fold(i_fold, valloader,model,device, criterion)
    val_results.append(val_fold_results)
    
    

print(f'Finished Training in {(time.perf_counter()-start):.2f} seconds')
train_results
val_results