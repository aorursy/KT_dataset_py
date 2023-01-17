# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 as cv

import os

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F

import torch.utils.data.dataloader as DataLoader

import torch.utils.data.sampler as sampler

import torchvision.transforms as transforms

import torch.nn as nn

import torch.optim as optim

from PIL import Image

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
torch.cuda.is_available()
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

target_dict = {0:"No DR(Diabetic Retinopathy)",

               1:"Mild",

               2:"Moderate",

               3:"Severe",

               4:"Proliferative DR(Diabetic Retinopathy)" }
class MyDataset(torch.utils.data.Dataset):

    def __init__(self,dir_img_path=None,dir_csv_path=None,Usage=None,transform=None):

        super().__init__()

        """

        Usage:

        """

        self.DIR_PATH = "../input"

        self.df = pd.read_csv(dir_csv_path)

        self.dir_img_path = dir_img_path

        self.images = self.loadImage(self.dir_img_path)

        self.Usage = Usage

        self.transform = transform

        print("{} Data length of image {}:".format(Usage,len(self.images)))

        print("{} Data length of csv file {}:".format(Usage,len(self.df)))

        

    def loadImage(self,path):

        return os.listdir(path) 

    

    def __getitem__(self,pos):

        obj = self.df.loc[pos]

        img_id = obj["id_code"]

        if self.Usage =="Training":

            label = obj["diagnosis"]

            

        img_id ="{}.png".format(img_id)

         

        img = cv.imread(os.path.join(self.dir_img_path,img_id))

        #img = np.moveaxis(img, -1, 0) # for shifting column in Tensor

        #print(img.shape)

        img = Image.fromarray(img)

        

        #sample = {'image': img, 'label': label}

        if self.transform:

            img = self.transform(img)

        #print(img.shape)

        if self.Usage == "Training":

            return img,label

        return img,obj["id_code"]

    def change_type(self,img,label):

        return img.astype(np.float32),label.astype(np.long)

    

    def read(self,image):

        return cv.imread(image)

    

    def reshape(self,image):

        return cv.resize(image,(244,244))

    

    def __len__(self):

        return len(self.df)

       

transformation  = transforms.Compose([transforms.Resize((224,224)),

                                     #transforms.Grayscale(num_output_channels=1),

                                     transforms.ColorJitter(0.1),

                                     transforms.RandomHorizontalFlip(),

                                     transforms.RandomVerticalFlip(),

                                     transforms.ToTensor(),

                                     transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])



myDataset = MyDataset("../input/train_images","../input/train.csv",transform=transformation,Usage="Training")

#trainingLoader = torch.utils.data.DataLoader(myDataset,batch_size=32,shuffle=True)

#myDataset[0]
test_size= 0.2

samples = len(myDataset)

indices = list(range(samples))

np.random.shuffle(indices)

train_len  =  int(np.floor(samples * (test_size)))

train_idx, valid_idx = indices[train_len:], indices[:train_len]

train_sampler = sampler.SubsetRandomSampler(train_idx)

valid_sampler = sampler.SubsetRandomSampler(valid_idx)

print(len(train_sampler),len(valid_sampler))



train_loader = torch.utils.data.DataLoader(myDataset,sampler= train_sampler

                                           ,batch_size = 32,shuffle=False)

test_loader = torch.utils.data.DataLoader(myDataset, sampler= valid_sampler,

                                          batch_size = 32,shuffle=False)

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
class Net(nn.Module):

    def __init__(self):

        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=7,padding=3)

        self.conv1_1 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=7,padding=3)

        

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=7,padding=3)

        self.conv2_1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=7,padding=3)

        

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding=2)

        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=5,padding=2)

        

        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5,padding=2)

        self.conv4_1 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,padding=2)

        

        self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=5,padding=2)

        self.conv5_1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=5,padding=2)

        

        self.fc1 = nn.Linear(in_features=512*7*7,out_features=1024)

        self.fc2 = nn.Linear(in_features=1024,out_features=1024)

        self.fc3 = nn.Linear(in_features=1024,out_features=512)

        self.out = nn.Linear(in_features=1024,out_features=5)

        

        self.dropout = nn.Dropout(0.5)

        

    def forward(self,x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv1_1(x))

        x = F.max_pool2d(x,kernel_size=2,stride=2)

        #print(x.shape)

        x = self.dropout(x)

        

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv2_1(x))

        x = F.max_pool2d(x,kernel_size=2,stride=2)

        #print(x.shape)

        x = self.dropout(x)

        

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv3_1(x))

        x = F.max_pool2d(x,kernel_size=2,stride=2)

        #print(x.shape)

        x = self.dropout(x)

        

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv4_1(x))

        x = F.max_pool2d(x,kernel_size=2,stride=2)

        #print(x.shape)

        x = self.dropout(x)

        

        x = F.relu(self.conv5(x))

        x = F.relu(self.conv5_1(x))

        x = F.max_pool2d(x,kernel_size=2,stride=2)

        x = self.dropout(x)

        #print(x.shape)

        x = x.reshape(-1,512*7*7)

        #print(x.shape)

        x = F.relu(self.fc1(x))

        

        x = self.dropout(x)

        #x = F.relu(self.fc2(x))

        #x = F.relu(self.fc3(x))

        #print(x.shape)

        x = F.log_softmax(self.out(x),dim=1)

        #print(x.shape)

        return x

net = Net().to(device)

#net.cuda()
criterion = torch.nn.NLLLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)

lr_step = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.3)
for epoch in range(20):

    running_loss = 0.0 

    valid_loss = []

    valid_auc = []

    train_loss = []

    train_auc = []

    #lr_step.step()

    net.train()

    for i,(image,label) in enumerate(train_loader): 

            image,label = image.to(device),label.to(device)

            optimizer.zero_grad()

            output = net(image)

            loss = criterion(output,label)

            _,output = torch.max(output,1)             

            loss.backward()

            optimizer.step()

            running_loss +=loss.item() 

            train_loss.append(loss.item())

            train_auc.append(accuracy_score(torch.Tensor.cpu(output),torch.Tensor.cpu(label)))

            if i % 10 == 9: # print every 10 mini-batches

                print('[%d, %5d] loss: %.5f Accuracy:%.5f' %(epoch + 1, i + 1,running_loss / 100,

accuracy_score(torch.Tensor.cpu(output),torch.Tensor.cpu(label))))

                running_loss = 0.0

    net.eval()  

    for i,(image,label) in enumerate(test_loader):

        image,label = image.to(device),label.to(device)

        output = net(image)

        loss = criterion(output,label)

        _,output = torch.max(output,1)   

        valid_loss.append(loss.item())

        valid_auc.append(accuracy_score(output.cpu().detach().numpy(),label.cpu().detach().numpy()))

    print('Epoch {}, train loss: {}, train accuracy: {}\tvalid loss: {}, valid accuracy: {}'.format(epoch+1,np.mean(train_loss),np.mean(train_auc),np.mean(valid_loss),np.mean(valid_auc)))

    #break
print("The state dict keys: \n\n", net.state_dict().keys())
checkpoint = {'model': Net(),

              'state_dict': net.state_dict(),

              'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
def load_model(filepath):

    checkpoint = torch.load(filepath)

    model = checkpoint['model']

    model.load_state_dict(checkpoint["state_dict"])

    for parameter in model.parameters():

        parameter.require_grad = False

    model.eval()

    return model
model = load_model("checkpoint.pth")
testData = MyDataset("../input/test_images","../input/test.csv",transform=transformation,Usage="Testing")
label_dict = {}

label_dict['id_code'] = list()

label_dict["diagnosis"] = list()

for image,tag in testData:

    image = image.unsqueeze(0)

    output = model(image)

    _,output = torch.max(output,1)

    label_dict["id_code"].append(tag)

    

    label_dict["diagnosis"].append(int(output))

    #print(int(output))

    #break
testFrame = pd.DataFrame(label_dict)

testFrame.to_csv("submission.csv")
testFrame