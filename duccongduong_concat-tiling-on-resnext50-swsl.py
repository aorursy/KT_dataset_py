import pandas as pd

import matplotlib.pyplot as plt



import cv2

import skimage.io



from sklearn.model_selection import train_test_split



import torch

import torch.nn as nn

from torch.utils.data import DataLoader, Dataset



import torchvision

from torchvision import models, transforms



from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip

from albumentations.pytorch import ToTensorV2



import torch.optim as optim

from torch.optim import lr_scheduler



import time

import copy



from PIL import Image
# Require compose objects

data_transforms = {

    'train': transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),

                                 transforms.RandomVerticalFlip(p=0.5),

                                 transforms.ToTensor(),

                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                      std=[0.229, 0.224, 0.225])]),

    'test': transforms.Compose([transforms.ToTensor(),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                     std=[0.229, 0.224, 0.225])])

}



def get_transforms(*, data):

    

    assert data in ('train', 'valid')

    

    if data == 'train':

        return Compose([

            HorizontalFlip(p=0.5),

            VerticalFlip(p=0.5),

            Normalize(

                mean=[0.485, 0.456, 0.406],

                std=[0.229, 0.224, 0.225],

            ),

            ToTensorV2(),

        ])

    

    elif data == 'valid':

        return Compose([

            Normalize(

                mean=[0.485, 0.456, 0.406],

                std=[0.229, 0.224, 0.225],

            ),

            ToTensorV2(),

        ])
class TrainDataset(Dataset):

    def __init__ (self, image_id, labels, transform=None):

        self.image_id = image_id

        self.labels = labels

        self.transform = transform

        

    def __len__(self):

        return len(self.image_id)

    

    def __getitem__(self, idx):

#         Creating the image

        name = self.image_id[idx]

        file_names = [f"../input/panda-16x128x128-tiles-data/train/{name}_{i}.png" for i in range(16)]

        image_tiles = skimage.io.imread_collection(file_names, conserve_memory=True)

        image_tiles = cv2.hconcat([cv2.vconcat([image_tiles[0], image_tiles[1], image_tiles[2], image_tiles[3]]),

                                   cv2.vconcat([image_tiles[4], image_tiles[5], image_tiles[6], image_tiles[7]]),

                                   cv2.vconcat([image_tiles[8], image_tiles[9], image_tiles[10], image_tiles[11]]),

                                   cv2.vconcat([image_tiles[12], image_tiles[13], image_tiles[14], image_tiles[15]])])

#         image_tiles = cv2.cvtColor(image_tiles, cv2.COLOR_BGR2RGB)

        if self.transform:

            image_tiles = Image.fromarray(image_tiles)

            image_tiles = self.transform(image_tiles)

#         Creating the label

        label = self.labels[idx]

        label = torch.tensor(label).float()

#         Return image, and label

        return image_tiles, label
class TestDataset(Dataset):

    def __init__ (self, image_id, dir_name=None, transform=None):

        self.image_id = image_id

        self.dir_name = dir_name

        self.transform = transform

        

    def __len__(self):

        return len(self.image_id)

    

    def __getitem__(self, idx):

#         Creating the image

        name = self.image_id[idx]

        file_names = [f"../panda-16x128x128-tiles-data/train/{name}_{i}.png" for i in range(16)]

        image_tiles = skimage.io.imread_collection(file_names, conserve_memory=True)

        image_tiles = cv2.hconcat([cv2.vconcat([image_tiles[0], image_tiles[1], image_tiles[2], image_tiles[3]]),

                                   cv2.vconcat([image_tiles[4], image_tiles[5], image_tiles[6], image_tiles[7]]),

                                   cv2.vconcat([image_tiles[8], image_tiles[9], image_tiles[10], image_tiles[11]]),

                                   cv2.vconcat([image_tiles[12], image_tiles[13], image_tiles[14], image_tiles[15]])])

#         image_tiles = cv2.cvtColor(image_tiles, cv2.COLOR_BGR2RGB)

        if self.transform:

            image_tiles = Image.fromarray(image_tiles)

            image_tiles = self.transform(image_tiles)

#         Return image

        return image_tiles 
def train_model(model, criterion, optimizer, scheduler, num_epochs=8, bs=5):

#     Time tracking and saving model weights and accuracies

    time_start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    

    train_dataset = TrainDataset(X_train, y_train, transform=data_transforms['train'])

    test_dataset = TrainDataset(X_test, y_test, transform=data_transforms['test'])

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    

    

    for epoch in range(num_epochs):

        print(f'Epoch {epoch}')

        

        for phase in ['train', 'test']:

            if phase == 'train':

                model.train()

                dataloader = train_loader

            else:

                model.eval()

                dataloader = test_loader

                

            running_loss = 0.0

            running_corrects = 0

            

            for inputs, labels in dataloader:

                inputs = inputs.to(device)

                labels = labels.to(device=device, dtype=torch.int64)

                

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):    

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                

                    if phase == 'train':

                        loss.backward(retain_graph=False)

                        optimizer.step()

                        

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

                

#             if phase == 'train':

#                 scheduler.step()

                

            if phase == 'train':

                epoch_loss = running_loss / len(train_dataset)

                epoch_acc = running_corrects.double() / len(train_dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            else:

                epoch_loss = running_loss / len(test_dataset)

                epoch_acc = running_corrects.double() / len(test_dataset)

                scheduler.step(epoch_acc)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                

            if phase == 'test' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

                

        print(f"Time: {(time.time()-time_start) // 60,} minutes and {(time.time()-time_start) % 60} seconds")

        print()

    time_taken = time.time() - time_start

    print(f'Training completed in {time_taken // 60,} minutes and {time_taken % 60} seconds')

    print(f"Best Val Accuracy: {best_acc}")

                

    model.load_state_dict(best_model_wts)

    return model 
class config:

    debug=False

    seed=101
train = pd.read_csv("../input/panda-train-csv-with-file-check/train_with_file_check.csv", index_col='Unnamed: 0')

print(train.shape)

train = train.query("has_file == True")

print(train.shape)



if config.debug:

    train = train.sample(150, random_state=config.seed, axis=0)

    

print(train.shape)



X = train['image_id']

y = train['isup_grade']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=config.seed)

# Reseting Index

X_train.reset_index(drop=True, inplace=True)

X_test.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)

y_test.reset_index(drop=True, inplace=True)
# Device (sending our model, inputs, labels to either cuda or cpu)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# Changing out_features

model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')

# model = models.resnet50(pretrained=True)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 6)

model = model.to(device)



# Other modules

# optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor=0.5, patience=3)



criterion = nn.CrossEntropyLoss()
# 10516, batch_size=16

train_model(model, criterion, optimizer_ft, exp_lr_scheduler, 

           num_epochs=25,

           bs=20)
# Saving Model

torch.save({

    'epoch': 25,

    'model_state_dict': model.state_dict(),

    'optimizer_state_dict': optimizer_ft.state_dict()

}, "../working/resnext_50_sw_10500_6hours_Adam_Plateua.tar")



torch.save(model, "resnext_50_sw_10500_6hours_Adam_Plateua.pth")