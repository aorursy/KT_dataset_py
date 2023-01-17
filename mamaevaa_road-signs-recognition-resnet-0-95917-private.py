import os
import numpy as np
import pandas as pd
import time
import copy
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torchvision
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import PIL
from PIL import Image
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
data_path = "/kaggle/input/road-signs-recognition/"
train_ann_path = os.path.join(data_path,'train.csv')
train_path = os.path.join(data_path, 'train/train/')
train_df = pd.read_csv(train_ann_path)
train, val = train_test_split(train_df, test_size=0.15, random_state=13, stratify=train_df.class_number)
train.reset_index(inplace=True, drop=True)
val.reset_index(inplace=True, drop=True)
num_classes = 67

batch_size = 32

num_epochs = 12

input_size = 224
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print(f"lr: {scheduler.optimizer.param_groups[0]['lr']}")
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval() 

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'): 
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def initialize_model(num_classes, feature_extract=True, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    for params in model_ft.layer4.parameters():
        params.requires_grad = True
        
    for params in model_ft.layer3.parameters():
        params.requires_grad = True
        
    return model_ft
model = initialize_model(num_classes)
class SignsDataset(data.Dataset):
    
    def __init__(self, root, flist, transform=None):
        """
        root: path to images
        imlist: pandas DataFrame with columns file_name, class
        transform: torchvision transform applied to every image
        """
        self.root  = root
        self.flist = flist 
        self.transform = transform
        
    def __getitem__(self, index):
        impath, target = self.flist.loc[index] 

        full_imname = os.path.join(self.root, impath)
        assert os.path.exists(full_imname), f'No file {full_imname}' 

        
        image = cv2.imread(full_imname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)["image"]
        
        return image, target, impath
    
    def __len__(self):
        return len(self.flist)
train_transformation = A.Compose([A.SmallestMaxSize(max_size=224),
                                  A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                                  A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                                  A.RandomBrightnessContrast(p=0.5),
                                  A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                  ToTensorV2()
                                 ])
        
val_transformation =  A.Compose([A.SmallestMaxSize(max_size=224),
                                 A.CenterCrop(height=224, width=224),
                                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                 ToTensorV2()
                                ])
train_dataset = SignsDataset(root=train_path, flist=train, transform=train_transformation)
val_dataset = SignsDataset(root=train_path, flist=val, transform=val_transformation)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)                          
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}
model = model.to(device)
# У всех параметров кроме переопределнных полносвязных слоев в функции initialize_model
# param.requires_grad выставлено в False методом set_parameter_requires_grad
params_to_update = [param for name, param in model.named_parameters() if param.requires_grad]
for name, param in model.named_parameters():
    if param.requires_grad: print(name)
optimizer = optim.Adam(params_to_update, lr=0.03)
criterion = nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=int(len(train_dataset)/batch_size + 1)*num_epochs)
model, hist = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs)
state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()
        }

torch.save(state, '/kaggle/working/resnet18.pth')
def prediction(model, dataloader):
    filename = []
    class_number = []
    for inputs, _, path in dataloader:
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        filename.extend(path)
        class_number.extend(preds.tolist())
            
    return pd.DataFrame(list(zip(filename, class_number)), columns=['filename', 'class_number'])
test_path =  os.path.join(data_path, 'test/test/')

test = pd.DataFrame(os.listdir(test_path), columns=['filename']) 
test['class_number'] = np.nan

test_dataset = SignsDataset(root=test_path, flist=test, transform=val_transformation)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
checkpoint = torch.load('/kaggle/working/resnet18.pth')
model_state_dict = checkpoint['state_dict']
model.load_state_dict(model_state_dict)
model = model.to(device)
result = prediction(model, test_dataloader)
result.head()
result.to_csv('/kaggle/working/resnet18sv6.csv', index=False)