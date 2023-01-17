import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms as T
from torchvision import datasets
import torchvision.models as models
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import matplotlib.pylab as plt
from PIL import Image
import cv2

import time
import os
import copy
import random
import math
plt.ion()   # interactive mode

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
os.listdir("../input/ammi-2020-convnets/train/train/cgm")[:5]
def setup_seed(seed, cuda):
    # Creates global random seed across torch, cuda and numpy 
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
from argparse import Namespace

args = Namespace(
    size = 224,

    # Model Hyperparameters
    learning_rate = 1e-4,
    batch_size = 16,
    num_epochs = 1,
    early_stopping_criteria=10,
    momentum=0.9,
    pretrained = True,
    
    # Data Parameters
    mean = torch.tensor([0.485, 0.456, 0.406]),
    std = torch.tensor([0.229, 0.224, 0.225]),
    shuffle_dataset = True,
    
    
    # Cv params
    num_folds=5,
    seed= 0,
    num_workers=4,
    
    # Paths
    train_path="../input/ammi-2020-convnets/train/train/",
    pretrained_model_path="../input/pretrained-pytorch-models/",
    save_dir='/kaggle/working/models',

    # Runtime hyper parameter
    cuda=True,

)

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")

print("Using CUDA: {}".format(args.cuda))

# Set seed for reproducibility
setup_seed(args.seed, args.cuda)
print('Train set:')

for cls in os.listdir(args.train_path):
    print('{} : {}'.format(cls, 
                         len(os.listdir(os.path.join(args.train_path, cls)))
                        ))
print()
im = Image.open(args.train_path + '/cgm/train-cgm-700.jpg')
print(im.size)
def get_labels(file_path): 
    dir_name = os.path.dirname(file_path)
    split_dir_name = dir_name.split("/")
    dir_levels = len(split_dir_name)
    label  = split_dir_name[dir_levels - 1]
    return(label)
from glob import glob
train_image_path = args.train_path + '*/*.*'
image_paths = glob(train_image_path, recursive=True)
image_paths[0:5]
images_df = pd.DataFrame(columns=['images', 'labels'])
images_df["images"] = image_paths

y = []
for img in image_paths:
    y.append(get_labels(img))   

images_df["labels"] = y
images_df.head()
labelencoder = LabelEncoder()

images_df["labels"] = labelencoder.fit_transform(images_df["labels"])
classes = labelencoder.classes_

print(classes)
args.num_classes = len(classes)
class CassavaDataset(Dataset):
    def __init__(self, df_data, transform=None):
        super().__init__()
        self.df = df_data.values
        
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path, label = self.df[index]
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)
        return image, label
train_trans = T.Compose([
        T.RandomResizedCrop(args.size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(args.mean, args.std)
    ])

val_trans = T.Compose([
        T.Resize(256),
        T.CenterCrop(args.size),
        T.ToTensor(),
        T.Normalize(args.mean, args.std)
    ])
def imshow(source, size):
    img, labels = source
    plt.figure(figsize=(10,10))
    for i in range(args.batch_size//2):
        plt.subplot(2, args.batch_size//2, i + 1)
        imt = (img[i].view(-1, size, size))
        imt = imt.numpy().transpose([1,2,0])
        imt = (args.std * imt + args.mean).clamp(0,1)
        plt.imshow(imt.squeeze(), cmap="gray")
        plt.title("{} ".format(classes[labels[i]]))
        
        plt.xticks([])
        plt.yticks([])
train_dataset = CassavaDataset(df_data=images_df, transform=train_trans)
train_loader = DataLoader(dataset = train_dataset, 
                          batch_size=args.batch_size, 
                          shuffle=args.shuffle_dataset, 
                          num_workers=args.num_workers)
source = next(iter(train_loader))
imshow(source, size=args.size)
cache_dir = "/root/.cache/torch/checkpoints/"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
!cp ../input/pretrained-pytorch-models/* $cache_dir
!ls $cache_dir
def prepare_model(pretrained=True):
    # We will use a pretrained model & model will be stored in a cache
    model_ft = models.resnet18(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, args.num_classes)

    model_ft = model_ft.to(args.device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), 
                             lr=args.learning_rate, 
                             momentum=args.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    return model_ft, criterion, optimizer_ft, exp_lr_scheduler
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in dataloaders:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
def stratified_kfold(num_folds=5, images_df=None):
    st_kfold = StratifiedKFold(n_splits=num_folds, shuffle=args.shuffle_dataset, random_state=args.seed)

    fold = 0
    for train_index, val_index in st_kfold.split(images_df['images'], images_df['labels']):
        train, val = images_df.iloc[train_index], images_df.iloc[val_index]

        train_dataset = CassavaDataset(df_data=train, transform=train_trans)
        valid_dataset = CassavaDataset(df_data=val,transform=val_trans)

        train_loader = DataLoader(dataset = train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=args.shuffle_dataset, 
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(dataset = valid_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=args.shuffle_dataset, 
                                  num_workers=args.num_workers)

        dataloaders = {'train': train_loader, 'val': valid_loader}
        
        dataset_sizes = {'train': len(train_dataset), 'val': len(valid_dataset)}
        print(dataset_sizes)

        print()
        print(f'Starting CV for Fold {fold}')
        model_ft, criterion, optimizer_ft, exp_lr_scheduler = prepare_model(pretrained=args.pretrained)
        model_ft = train_model(model_ft, 
                              criterion, 
                              optimizer_ft, 
                              exp_lr_scheduler,
                              dataloaders,
                              dataset_sizes,
                              num_epochs=args.num_epochs,)
        
        # Save model for the current fold to your output directory
        current_fold_full_path = args.save_dir +'/model_'+str(fold)+'_tar'
        torch.save(model_ft.state_dict(), current_fold_full_path)
        
        fold += 1
stratified_kfold(args.num_folds, images_df)
def train_whole_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in dataloaders:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
model_ft, criterion, optimizer_ft, exp_lr_scheduler = prepare_model(pretrained=args.pretrained)

train_dataset = CassavaDataset(df_data=images_df, transform=train_trans)
train_loader = DataLoader(dataset = train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_dataset, num_workers=args.num_workers)

dataloaders = {'train': train_loader}

dataset_sizes = {'train': len(train_dataset)}

model_ft = train_whole_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=args.num_epochs)

print('Training Done ...')
# model_path = args.save_dir +'/model_'+str(fold)+'_tar'
# model_ft = torch.load(model_path)