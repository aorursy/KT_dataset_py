import os

import numpy as np

import pandas as pd

import torch

from torchvision import transforms, datasets

from torch.utils.data import DataLoader, Dataset

from PIL import Image

import matplotlib.pyplot as plt

device = "cuda"
import torch.nn as nn

import torch.nn.functional as F



class Sq_Ex_Block(nn.Module):

    def __init__(self, in_ch, r):

        super(Sq_Ex_Block, self).__init__()

        self.se = nn.Sequential(

            GlobalAvgPool(),

            nn.Linear(in_ch, in_ch//r),

            nn.ReLU(inplace=True),

            nn.Linear(in_ch//r, in_ch),

            nn.Sigmoid()

        )



    def forward(self, x):

        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)

        return x.mul(se_weight)



class GlobalAvgPool(nn.Module):

    def __init__(self):

        super(GlobalAvgPool, self).__init__()

    def forward(self, x):

        return x.view(*(x.shape[:-2]),-1).mean(-1)



class SE_Net(nn.Module):

    def __init__(self,in_channels):

        super(SE_Net,self).__init__()

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 

        #                dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64,kernel_size=3,stride=1,padding=0)

        self.bn1 = nn.BatchNorm2d(num_features=64,eps=1e-3,momentum=0.01)

        self.c2 = nn.Conv2d(64,64,3,1,0)

        self.bn2 = nn.BatchNorm2d(64,1e-3,0.01)

        self.c3 = nn.Conv2d(64,64,5,1,2)

        self.bn3 = nn.BatchNorm2d(64,1e-3,0.01)

        self.m1 = nn.MaxPool2d(2)

        self.d1 = nn.Dropout(0.4)

        

        self.c4 = nn.Conv2d(64,128,3,1,0)

        self.bn4 = nn.BatchNorm2d(128,1e-3,0.01)

        self.c5 = nn.Conv2d(128,128,3,1,0)

        self.bn5 = nn.BatchNorm2d(128,1e-3,0.01)

        self.c6 = nn.Conv2d(128,128,5,1,2)

        self.bn6 = nn.BatchNorm2d(128,1e-3,0.01)

        self.m2 = nn.MaxPool2d(2)

        self.d2 = nn.Dropout(0.4)

        

        self.c7 = nn.Conv2d(128,256,3,1,0)

        self.bn7 = nn.BatchNorm2d(256,1e-3,0.01)

        self.se3 = Sq_Ex_Block(in_ch=256,r=8)

        self.m3 = nn.MaxPool2d(2)

        self.d3 = nn.Dropout(0.4)



        self.fc1 = nn.Linear(256*1*1,256)

        self.bn8 = nn.BatchNorm1d(256,1e-3,0.01)

        

        self.out = nn.Linear(256,10)

        

        self.init_linear_weights()

        

    def forward(self,x):

        x = self.bn1(F.leaky_relu(self.c1(x),0.1))

        x = self.bn2(F.leaky_relu(self.c2(x),0.1))

        x = self.bn3(F.leaky_relu(self.c3(x),0.1))

        x = self.d1(self.m1(x))

        

        x = self.bn4(F.leaky_relu(self.c4(x),0.1))

        x = self.bn5(F.leaky_relu(self.c5(x),0.1))

        x = self.bn6(F.leaky_relu(self.c6(x),0.1))

        x = self.d2(self.m2(x))

        

        x = self.bn7(F.leaky_relu(self.c7(x),0.1))

        x = self.se3(x)

        x = self.d3(self.m3(x))

        

        x = x.view(-1, 256*1*1) #reshape

        x = self.bn8(F.leaky_relu(self.fc1(x),0.1))

        return self.out(x)

    

    def init_linear_weights(self):

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in')  #default mode: fan_in

        nn.init.kaiming_normal_(self.out.weight, mode='fan_in')
trans = transforms.Compose([

        transforms.RandomAffine(degrees=10,translate=(0.15,0.15),scale=[0.9,1.1],shear=5), #Data augmentation

        transforms.ToTensor(),  #Take Image as input and convert to tensor with value from 0 to1  

    ])

trans_val = transforms.Compose([

        transforms.ToTensor(),  #Take Image as input and convert to tensor with value from 0 to1

    ])

trans_test = transforms.Compose([

        transforms.ToTensor(),  #Take Image as input and convert to tensor with value from 0 to1

    ])
global_data = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

global_data_test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")



class KMnistDataset(Dataset):

    def __init__(self,data_len=None, is_validate=False,validate_rate=None,indices=None, data=None):

        self.is_validate = is_validate

        self.data = global_data

        if data_len == None:

            data_len = len(self.data)

        

        self.indices = indices

        if self.is_validate:

            self.len = int(data_len*validate_rate)

            self.offset = int(data_len*(1-validate_rate))

            self.transform = trans_val

        else:

            self.len = int(data_len*(1-validate_rate))

            self.offset = 0

            self.transform = trans

        

    def __getitem__(self, idx):

        idx += self.offset

        idx = self.indices[idx]

        img = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape((28, 28))  #value: 0~255

        label = self.data.iloc[idx, 0]  #shape: (num,)

        img = Image.fromarray(img)

        img = self.transform(img)     #value: 0~1, shape:(1,28,28)

        label = torch.as_tensor(label, dtype=torch.uint8)    #value: 0~9, shape(1)

        return img, label



    def __len__(self):

        return self.len

    

class TestDataset(Dataset):

    def __init__(self,data_len=None):

        self.data = global_data_test

        self.transform = trans_test

        if data_len == None:

            self.len = len(self.data)

        

    def __getitem__(self, idx):

        img = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape((28, 28))  #value: 0~255

        img = Image.fromarray(img)

        img = self.transform(img)     #value: 0~1, shape:(1,28,28)

        return img, torch.Tensor([])



    def __len__(self):

        return self.len
batch_size = 1024

num_workers = 8

epochs = 70

lr = 1e-3

val_period = 1

val_rate = 0.1    ###Train->54000 images, Validation->6000 images
model = SE_Net(in_channels=1)

if device == "cuda":

    model.cuda()

    

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.99))

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15,factor=0.1)
indices_len = len(global_data)  ###For dataset

indices = np.arange(indices_len)



train_dataset = KMnistDataset(data_len=None,is_validate=False, validate_rate=val_rate,indices=indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)



val_dataset = KMnistDataset(data_len=None,is_validate=True, validate_rate=val_rate, indices=indices)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
min_loss = 10000

max_acc = 0

best_model_dict = None



print("Start training...")

for ep in range(0,epochs+1):

    model.train()

    data_num = 0

    

    ###Train

    for idx, data in enumerate(train_loader):

        img, target = data

        img, target = img.to(device), target.to(device,dtype=torch.long)



        pred = model(img)

        loss = criterion(pred,target)

        data_num += img.size(0)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



    ###Validate

    if ep!=0 and ep%val_period == 0:

        model.eval()

        acc = 0

        val_loss = 0

        data_num  = 0

        with torch.no_grad():

            for idx, data in enumerate(val_loader):

                img, target = data

                img, target = img.to(device), target.to(device,dtype=torch.long)

                

                pred = model(img)

                val_loss += criterion(pred, target).item()

                _,pred_class = torch.max(pred.data, 1)

                acc += (pred_class == target).sum().item()

                data_num += img.size(0)

        

        acc /= data_num

        val_loss /= data_num



        ###Reduce learning rate and Early stopping

        lr_scheduler.step(val_loss)

        if optimizer.param_groups[0]['lr'] < 1e-4:

            break                    



        ###Save the best model

        if acc >= max_acc:

            max_acc = acc

            min_loss = val_loss

            best_model_dict = model.state_dict()                    



        print("Episode:{}, Validation Loss:{},Accuracy:{:.4f}%,learning rate:{}"

              .format(ep,val_loss,acc*100,optimizer.param_groups[0]['lr']))

    

print("===================Best Model, Loss:{} Accuracy:{}==================".format(min_loss,max_acc))

print("====================================================================")

torch.cuda.empty_cache()
result = np.array([],dtype=np.int)

test_dataset = TestDataset(data_len=None)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)



test_model = SE_Net(in_channels=1)

test_model.load_state_dict(best_model_dict)

if device == "cuda":

    test_model.cuda()

test_model.eval()



with torch.no_grad():

    for idx, data in enumerate(test_loader):

        img = data[0].to(device)

        pred = test_model(img)

        _,pred_class = torch.max(pred.data, 1)

        result = np.concatenate([result,pred_class.cpu().numpy()],axis=0)

print("shape of the result:",np.shape(result))
sample_sub=pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

sample_sub['label']=result

sample_sub.to_csv('submission.csv',index=False)

sample_sub.head()