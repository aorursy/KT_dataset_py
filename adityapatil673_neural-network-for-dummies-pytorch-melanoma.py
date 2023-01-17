!pip install efficientnet_pytorch torchtoolbox
import pandas as pd
import numpy as np
import cv2
from efficientnet_pytorch import EfficientNet
import torchtoolbox.transform as transforms
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils import data

import matplotlib.pyplot as plt
%matplotlib inline

!pip install torchsummary
from torchsummary import summary

from torch.utils.data import Dataset, DataLoader
class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, transforms_object= None):
        self.data_frame = df
        self.path_to_folder = imfolder
        self.transforms_object = transforms_object
        
    def __getitem__(self, index):
        image_name_at_index = self.data_frame.loc[index,'image_name']
        load_path = self.path_to_folder +image_name_at_index+'.jpg'
        image_data = cv2.imread(load_path)
        if self.transforms_object:
            image_data = self.transforms_object(image_data)
        if 'target' in self.data_frame.columns.values:
            y = self.data_frame.loc[index,'target']
        else :
            y = 1
        return image_data,y
        
    def __len__(self):
        return self.data_frame.shape[0]


class deeper_network(nn.Module):
    def __init__(self,arch):
        super(deeper_network,self).__init__()
        self.arch = arch
        self.arch._fc = nn.Linear(in_features=1280,out_features=512, bias=True)
        self.fin_net = nn.Sequential(self.arch,
                                     nn.Linear(512,128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128,16),
                                     nn.LeakyReLU(),
                                     nn.Linear(16,1))
    def forward(self,inputs):
        output = self.fin_net(inputs)
        return output
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train_transform = transforms.Compose([
                                    transforms.RandomResizedCrop(size=256,scale=(0.7,1)), # Take 70 - 100 % of the area and scale the image to 256 x 256 size
                                    transforms.RandomHorizontalFlip(),# Take the image and flip it horizontally or not 50% chance
                                    transforms.RandomVerticalFlip(), # Take the image and flip it vertically or not 50% chance
                                    transforms.ToTensor(), # Convert to tensor
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # Adjust the values of image to standardise
model_eff = EfficientNet.from_pretrained('efficientnet-b1')
# print(summary(model_eff,(3, 256, 256)))
path_for_jpeg= '../input/siim-isic-melanoma-classification/jpeg/train/'
train_dataset = MelanomaDataset(train_df,path_for_jpeg,transforms_object=train_transform)
train_loader_args = dict(shuffle=True, batch_size=64)
train_loader = data.DataLoader(train_dataset, **train_loader_args)

arch = EfficientNet.from_pretrained('efficientnet-b1') 

#setup
deep_net = deeper_network(arch)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(deep_net.parameters())

# Sometimes, I just use my previously trained network and load it just before training again
deep_net.load_state_dict(torch.load('../input/effnet-v2/effnet_v2',map_location=torch.device('cpu'))) 
torch.cuda.is_available()
use_cuda = True
if use_cuda and torch.cuda.is_available():
    deep_net.cuda() # converting the model into GPU enabled variable
model = deep_net
import time
model.train()
for e in range(2,4):
    
    # variables to log results
    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    start_time = time.time()
    
    #loop for running the training in batches
    for batch_idx, (image_data_array, target) in enumerate(train_loader):
        
        #setting up batch data 
        optimizer.zero_grad()   # .backward() accumulates gradients
        image_data_array = image_data_array.float().cuda()
        target = target.long().cuda() # all data & model on same device
        
        #Prediction
        outputs = model(image_data_array)

        #Measuring the Error
        loss = criterion(outputs, target.reshape(-1,1).float())
        
        #Logging Error
        predictions = torch.round(torch.sigmoid(outputs))
        total_predictions += target.size(0)
        correct_predictions += (target.cpu() ==predictions.squeeze().cpu()).sum().item()
        running_loss += loss.item()
        
        #Correcting the model to reduce the error
        loss.backward()
        optimizer.step()
    
    acc = (correct_predictions/total_predictions)*100.0
    end_time = time.time()

    running_loss /= len(train_loader)
    print('Training Loss: ', round(running_loss,3), 'Time: ',round(end_time - start_time,3), 's')
    print('Training Accuracy: ', round(acc,3), '%')
    
torch.save(model.state_dict(), 'effnet_v1')
path_for_jpeg= '../input/siim-isic-melanoma-classification/jpeg/test/'
test_dataset = MelanomaDataset(test_df,path_for_jpeg,transforms_object=train_transform)
test_loader_args = dict(shuffle=False, batch_size=10) # DONT SHUFFLE
test_loader = data.DataLoader(test_dataset, **test_loader_args)
model.eval()
fin_temp=np.empty((0,))
for batch_idx, (image_data_array, target) in enumerate(test_loader):
    image_data_array = image_data_array.float()#.cuda()
    target = target.long()#.cuda() # all data & model on same device
    outputs = model(image_data_array)
    temp = torch.sigmoid(outputs).cpu().detach().numpy().squeeze()
    fin_temp = np.concatenate([fin_temp,temp])
Y_submission = test_df[['image_name']].copy()
Y_submission['target'] = fin_temp
Y_submission.to_csv('/kaggle/working/image_v3.csv',index=False)
