import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, exists, isdir
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, datasets, models
import torchvision.models as models
import torch.optim.lr_scheduler as schedule
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
!ls ../input/mnist-but-chinese/MNIST_Chinese_Hackathon
train = pd.read_csv('/kaggle/input/mnist-but-chinese/MNIST_Chinese_Hackathon/train.csv')
test = pd.read_csv('/kaggle/input/mnist-but-chinese/MNIST_Chinese_Hackathon/test.csv')
sample = pd.read_csv('/kaggle/input/mnist-but-chinese/MNIST_Chinese_Hackathon/sample_submission.csv')
df_tr,df_val = train_test_split(train,random_state=42,test_size=0.2)
print('Training dataset shape ',df_tr.shape)
print('Validation dataset shape ',df_val.shape)
import seaborn as sns
sns.set_style('whitegrid')
plt.figure(figsize=(9,5))
(train['code'].value_counts()/len(train)*100).plot(kind='bar')
plt.title('Training set distribution in %')
plt.show()
fig, ax = plt.subplots(1,2,figsize=(18,5))
print('Comparison between Training and Validation set')
(df_tr['code'].value_counts()/len(df_tr)*100).sort_index().plot(kind='bar',ax=ax[0],title='training set distribution')
(df_val['code'].value_counts()/len(df_val)*100).sort_index().plot(kind='bar',ax=ax[1],title='validation set distribution')
plt.show()
class builddataset(Dataset):
    def __init__(self,data,root,transforms=None):
        self.data = data 
        self.root = root
        self.transforms = transforms
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        filename = self.root+'/'+str(self.data.iloc[idx,0])
        image = Image.open(filename)
        label = torch.tensor(np.array(self.data.iloc[idx,1]-1))
        if self.transforms:
            image = self.transforms(image)
        return (image,label)
from PIL import Image, ImageOps, ImageEnhance
import numbers

class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift
        
    @staticmethod
    def get_params(shift):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift 
    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)
        
        return img.transform(img.size, Image.AFFINE, (1,0,hshift,0,1,vshift), resample=Image.BICUBIC, fill=1)
train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                       RandomShift(6),
                                       transforms.Resize(64,64),
                                       transforms.RandomResizedCrop(64),
                                       transforms.ToTensor()])

val_transforms = transforms.Compose([transforms.Resize(64,64),
                                     transforms.ToTensor()])
data_dir = '../input/mnist-but-chinese/MNIST_Chinese_Hackathon'
# Training and Validation Datasets
train_ds = builddataset(df_tr,data_dir+'/Training_Data',transforms=train_transforms)
val_ds = builddataset(df_val,data_dir+'/Training_Data',transforms=val_transforms)

# Train and validation Dataloaders
train_dl = DataLoader(train_ds,batch_size=32,shuffle=True)
val_dl = DataLoader(val_ds,batch_size=32,shuffle=False)

dataloader = {'train':train_dl,'val':val_dl}
import time
import copy
def train_model(model,criterion,optimizer,scheduler,num_epochs):
    since = time.time()
    best_model_p = copy.deepcopy(model.state_dict())
    best_accu = 0.0
    for epoch in range(1,num_epochs+1):
        if epoch%10 == 0:
            print('Epoch: {}/{}'.format(epoch,num_epochs))
        for phase in ['train','val']:
            if(phase=='train'):
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            for inputs,targets in dataloader[phase]:
                inputs=inputs.cuda()
                targets=targets.cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    targets = torch.flatten(targets)
                    loss = criterion(outputs,targets)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == targets.data).double().item()
            if phase=='train':
                epoch_loss = running_loss/len(train_ds)
                epoch_accu = running_corrects/len(train_ds)
            else:
                epoch_loss = running_loss/len(val_ds)
                epoch_accu = running_corrects/len(val_ds)
                scheduler.step(epoch_loss)
            if epoch%10 == 0:
                print(phase,' Loss: ',epoch_loss,' Accuracy: ',epoch_accu)
            if phase=='val' and epoch_accu >= best_accu:
                best_accu = epoch_accu
                best_model_p = copy.deepcopy(model.state_dict())
    time_taken = time.time()-since
    print("Training Completed in time ",time_taken)
    print("Best Accuracy ",best_accu)
    model.load_state_dict(best_model_p)
    return model
import torch.nn as nn
import torch.nn.functional as F
class cnn_model1(nn.Module):
    def __init__(self):
        super(cnn_model1,self).__init__()
        
        self.Conv1 = nn.Conv2d(1,32,3,1,1) 
        self.Bn2d1 = nn.BatchNorm2d(32)
        
        self.Conv2 = nn.Conv2d(32,32,3,1,1) 
        self.Bn2d2 = nn.BatchNorm2d(32)
        self.Max2 = nn.MaxPool2d(2,2)   
        
        self.Conv3 = nn.Conv2d(32,64,3,1,1)
        self.Bn2d3 = nn.BatchNorm2d(64)
        self.Max3 = nn.MaxPool2d(2,2)       
        
        self.Conv4 = nn.Conv2d(64,64,3,1,1) 
        self.Bn2d4 = nn.BatchNorm2d(64)
        self.Max4 = nn.MaxPool2d(2,2)   
        
        
        self.fc1 = nn.Linear(64*8*8,512)
        self.Bn1d1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(512,512)
        self.Bn1d2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512,15)
    
    def forward(self,x):
        x = F.relu(self.Bn2d1(self.Conv1(x)))
        x = self.Max2(F.relu(self.Bn2d2(self.Conv2(x))))
        x = self.Max3(F.relu(self.Bn2d3(self.Conv3(x))))
        x = self.Max4(F.relu(self.Bn2d4(self.Conv4(x))))
        x = x.view(x.size(0),-1)
        x = self.drop1(F.relu(self.Bn1d1(self.fc1(x))))
        x = self.drop2(F.relu(self.Bn1d2(self.fc2(x))))
        x = self.fc3(x)
        return x
model1 = cnn_model1()
model1.cuda()
print('model1 in GPU')
criterion1 = torch.nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(),lr=0.003)
scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1,patience=5,factor=0.2,min_lr=0.0000001)
trained_model1 = train_model(model1,criterion1,optimizer1,scheduler1,num_epochs=40)
class cnn_model2(nn.Module):
    def __init__(self):
        super(cnn_model2,self).__init__()
        
        self.Conv1 = nn.Conv2d(1,32,3,1,1)  
        self.bn1 = nn.BatchNorm2d(32)
        self.Conv2 = nn.Conv2d(32,32,3,1,1) 
        self.bn2 = nn.BatchNorm2d(32)
        self.Conv3 = nn.Conv2d(32,32,4,2,1) 
        self.bn3 = nn.BatchNorm2d(32)
        self.drop3 = nn.Dropout(p=0.4)
        self.Conv4 = nn.Conv2d(32,64,3,1,1) 
        self.bn4 = nn.BatchNorm2d(64)
        self.Conv5 = nn.Conv2d(64,64,3,1,1)
        self.bn5 = nn.BatchNorm2d(64)
        self.Conv6 = nn.Conv2d(64,64,4,2,1)
        self.bn6 = nn.BatchNorm2d(64)
        self.drop6 = nn.Dropout(p=0.4)
        self.Conv7 = nn.Conv2d(64,64,3,1,1)
        self.bn7 = nn.BatchNorm2d(64)
        
        self.drop_fc = nn.Dropout(p=0.4)
        self.fc = nn.Linear(64*16*16,15)
        
    def forward(self,x):
        x=F.relu(self.bn1(self.Conv1(x)))
        x=F.relu(self.bn2(self.Conv2(x)))
        x=self.drop3(F.relu(self.bn3(self.Conv3(x))))
        x=F.relu(self.bn4(self.Conv4(x)))
        x=F.relu(self.bn5(self.Conv5(x)))
        x=self.drop6(F.relu(self.bn6(self.Conv6(x))))
        x=F.relu(self.bn7(self.Conv7(x)))
        x=x.view(x.size(0),-1)
        x=self.drop_fc(x)
        x=self.fc(x)
        return x
model2 = cnn_model2()
model2.cuda()
print('model2 in GPU')
criterion2 = torch.nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(model2.parameters(),lr=0.003)
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2,patience=5,factor=0.2,min_lr=0.0000001)
trained_model2 = train_model(model2,criterion2,optimizer2,scheduler2,num_epochs=40)
def evaluate(model):
    model.eval()
    running_loss = 0
    running_corrects = 0
    for inputs,targets in dataloader['val']:
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)
        _,preds = torch.max(outputs,1)
        targets = torch.flatten(targets)
        running_corrects += torch.sum(preds == targets.data)
    print("Accuracy: ",running_corrects.item()/len(val_ds))
    return
print('Accuracy score on the validation set using CNN Model 1: ')
evaluate(trained_model1)

print('Accuracy score on the validation set using CNN Model 2: ')
evaluate(trained_model2)
class ensemble_average(nn.Module):
    def __init__(self,model1,model2):
        super(ensemble_average,self).__init__()
        self.model1 = model1
        self.model2 = model2
    def forward(self,x):
        self.model1.eval()
        for params in self.model1.parameters():
            params.requires_grad = False
        self.model2.eval()
        for params in self.model2.parameters():
            params.requires_grad = False
        out1 = self.model1(x)
        out2 = self.model2(x)
        out = (out1+out2)/2
        return out
ensembled_model = ensemble_average(trained_model1,trained_model2)
print('Accuracy score on the validation set using ensembled(Average) model: ')
evaluate(ensembled_model)
# Building custom dataset for the test data

class buildtestdataset(Dataset):
    def __init__(self,data,root,transforms=None):
        self.data = data
        self.root = root
        self.transforms = transforms
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        filename = self.root+'/'+str(self.data.iloc[idx,0])
        image = Image.open(filename)
        if self.transforms:
            image = self.transforms(image)
        return (image)
def create_submission(model):
    test_ds = buildtestdataset(test,data_dir+'/Testing_Data',transforms=transforms.Compose([transforms.Resize(64,64),transforms.ToTensor()]))
    test_dl = DataLoader(test_ds,batch_size=1,shuffle=False)
    predictions = []
    for inputs in test_dl:
        inputs = inputs.cuda()
        outputs = model(inputs)
        _,preds = torch.max(outputs,1)
        preds = preds.item()
        predictions.append(preds)
    sub = pd.DataFrame(zip(test.id,predictions),columns=['id','code'])
    sub['code'] = sub['code']+1
    sub.set_index('id',inplace=True)
    return sub
sub = create_submission(ensembled_model)
sub.head()
sub.to_csv('final_submission.csv')