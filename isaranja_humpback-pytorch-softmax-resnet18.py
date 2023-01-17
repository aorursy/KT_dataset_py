import os

#import PIL
#print(PIL.PILLOW_VERSION)

import pandas as pd
import numpy as np
#import time
#import json
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

from torch.utils.data import TensorDataset, DataLoader,Dataset

from collections import OrderedDict
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import os
print(os.listdir("../input"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_df = pd.read_csv("../input/train.csv")
le = LabelEncoder()
train_df['target'] = le.fit_transform(train_df['Id'])
train_df.head()
class HW_Dataset(Dataset):
    def __init__(self,filepath,train_df,transform=None):
        self.file_path = filepath
        self.df = train_df
        self.transform = transform
        self.image_list = [x for x in os.listdir(self.file_path)]
        
    def __len__(self):
        return(len(self.image_list))
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.file_path,self.df.Image[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        target = self.df.target[idx]
        return (img,target)
transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                transforms.RandomHorizontalFlip(), 
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
train_dataset = HW_Dataset(filepath='../input/train/',train_df=train_df,transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=0)
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('do1', nn.Dropout(0.2)),
                          ('fc1', nn.Linear(512, 5005))
                          ]))
    
model.fc = classifier

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
n_epochs = 30
valid_loss_min = np.Inf
#best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(1, n_epochs+1):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    train_acc = 0.0
    valid_acc = 0.0
    ###################
    # train the model #
    ###################
    model.train()
    
    scheduler.step()
    running_loss = 0.0
    running_corrects = 0

    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        #print(target.data)
        data, target = data.to(device), target.to(device)
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        _, preds = torch.max(output, 1)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        running_loss += loss.item() * data.size(0)
        running_corrects += torch.sum(preds == target.data)
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    ######################    
    # validate the model #
    ######################
    #model.eval()
    #for data, target in data_loader['valid']:
    #    # move tensors to GPU if CUDA is available
    #    data, target = data.to(device), target.to(device)
    #    # forward pass: compute predicted outputs by passing inputs to the model
    #    output = model(data)
    #    _, preds = torch.max(output, 1)
    #    # calculate the batch loss
    #    loss = criterion(output, target)
    #    # update average validation loss 
    #    valid_loss += loss.item()*data.size(0)
    #    valid_acc += torch.sum(preds == target.data)
    
    # calculate average losses
    #train_loss = train_loss/len(data_loader['train'].dataset)
    #valid_loss = valid_loss/len(data_loader['valid'].dataset)
    
    #train_acc = train_acc.double()/len(data_loader['train'].dataset)
    #valid_acc = valid_acc.double()/len(data_loader['valid'].dataset)
    
    # print training/validation statistics 
    #print('Epoch: {} \t{:.6f} \t {:.6f} \t {:.0%} \t {:.0%}'.format( epoch, train_loss, valid_loss, train_acc, valid_acc))
    print('Epoch: {} \t{:.6f} \t {:.0%}'.format( epoch, epoch_loss, epoch_acc))
    # save model if validation loss has decreased
    #if valid_loss <= valid_loss_min:
    #    print('Validation loss decreased ({:.6f} --> {:.6f}).  copying model weights ...'.format(valid_loss_min,valid_loss))
    #    #best_model_wts = copy.deepcopy(model.state_dict())
    #    valid_loss_min = valid_loss
        
#model.load_state_dict(best_model_wts)
