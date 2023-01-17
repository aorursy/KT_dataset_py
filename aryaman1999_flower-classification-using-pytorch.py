# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from torchvision import datasets,transforms
#root_dir = "/kaggle/input/flowers-recognition/flowers/flowers"
root_dir = "/kaggle/input/final-flowers-course-project-dataset/newFlowers"
flower_transform = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
flower_dataset = datasets.ImageFolder(root_dir,transform=flower_transform)
### Train Test Parameters ####

batch_size = 8
valid_split = 0.2
shuffle_dataset = True
random_seed = 42
flower_dataset.classes
### Split dataset into train and test #####
import numpy as np 
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
dataset_size = len(flower_dataset)
indices = list(range(dataset_size))
split = int(np.floor(valid_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices,val_indices = indices[split:],indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(flower_dataset,batch_size=batch_size,sampler=train_sampler)
valid_loader = DataLoader(flower_dataset,batch_size=batch_size,sampler=valid_sampler)



#### Visualize Images in Train and Test #####
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

train_batch,label_train = next(iter(train_loader))
valid_batch,label_val = next(iter(valid_loader))





def img_plotter(batch,rows=8,cols=8):
    fig,axs = plt.subplots(nrows=rows,ncols=cols,figsize=(30,30))
    for i in range(rows):
        for j in range(cols):
            axs[i,j].imshow(batch[rows*i+j].permute(1,2,0))
        

train_batch,label_train = next(iter(train_loader))
valid_batch,label_val = next(iter(valid_loader))

### Visualize training images ####
img_plotter(train_batch,rows=4,cols=4)
####Visualize validation images ####
img_plotter(valid_batch,rows=4,cols=4)
##### Design Custom Network ######
import torch.nn as nn
'''class Flower_Net(nn.Module):
    def __init__(self):
        
        super(Flower_Net,self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(3,64,kernel_size=5,padding=2),
        nn.BatchNorm2d(64),
        nn.Dropout(0.8),
        nn.MaxPool2d(kernel_size=3,padding=1),
        
        )
        self.layer2 = nn.Sequential(
        nn.Conv2d(64,64,kernel_size=5,padding=2),
        nn.Dropout(0.8),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=3,padding=1),
        
        )
        self.layer3 = nn.Sequential(
        nn.Conv2d(64,32,kernel_size=3,padding=1),
        nn.BatchNorm2d(32),
        nn.Dropout(0.8),
        nn.MaxPool2d(kernel_size=3,padding=1),
        )
        self.layer4 = nn.Sequential(
        nn.Conv2d(32,32,kernel_size=3,padding=1),
        nn.Dropout(0.8),
        nn.BatchNorm2d(32),
        #nn.MaxPool2d(kernel_size=3,padding=1),
        )
        self.layer5 = nn.Sequential(
        nn.Conv2d(32,16,kernel_size=3,padding=1),
        nn.BatchNorm2d(16),
        nn.Dropout(0.8),
        nn.MaxPool2d(kernel_size=3,padding=1),
        )
        self.layer6 = nn.Sequential(
        nn.Conv2d(16,8,kernel_size=3,padding=1),
        nn.BatchNorm2d(8),
        nn.Dropout(0.8),
        nn.MaxPool2d(kernel_size=3,padding=1),
        )
        self.flat = nn.Flatten()
        self.fc = nn.Linear(72,5)
    
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.flat(x))
        out = self.fc(x)
        return out
    

class Flower_Net(nn.Module):
    def __init__(self):
        super(Flower_Net,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,16,kernel_size=3,padding=1),nn.AvgPool2d(kernel_size=3,padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(16,8,kernel_size=3,padding=1),nn.MaxPool2d(kernel_size=3,padding=1))
        
        self.layer3 = nn.Dropout(0.5)
        
        self.layer4 = nn.Sequential(nn.Conv2d(8,16,kernel_size=3,padding=1))
        
        self.layer5 = nn.Flatten()
        self.layer6 = nn.Linear(51984,3000)
        self.layer7 = nn.Dropout(0.4)
        self.layer8 = nn.Linear(3000,64)
        self.layer9 = nn.Linear(64,5)
        self.layer10 = nn.LogSoftmax(dim=1)
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        out = self.layer9(x)
        #out = self.layer10(x)
        return out'''
        

class Flower_Net(nn.Module):
    def __init__(self):
        super(Flower_Net,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,16,kernel_size=3,padding=1),
                                    nn.Conv2d(16,8,kernel_size=3,padding=1),
                                    nn.AvgPool2d(kernel_size=3,padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(8,8,kernel_size=3,padding=1),
                                    nn.Conv2d(8,4,kernel_size=3,padding=1)
                                    ,nn.MaxPool2d(kernel_size=3,padding=1))
        self.layer3 = nn.Flatten()
        self.layer4 = nn.Linear(12996,512)
        self.layer5 = nn.Dropout(0.7)
        
        self.layer6 = nn.Linear(512,5)
        
    def forward(self,x):
        x = self.layer1(x)
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        out = self.layer6(x)
        return out
        
            
model = Flower_Net()
model
    
        
#### Ensemble ###
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class Flower_Net_1(nn.Module):
    def __init__(self):
        super(Flower_Net_1,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,128,kernel_size=3,padding=1),nn.AvgPool2d(kernel_size=3,padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(128,64,kernel_size=3,padding=1),nn.MaxPool2d(kernel_size=5,padding=2))
        self.layer3 = nn.Sequential(nn.Conv2d(64,32,kernel_size=3,padding=1),nn.MaxPool2d(kernel_size=5,padding=2))
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        out = self.flatten(x)
        return out

class Flower_Net_2(nn.Module):
    def __init__(self):
        super(Flower_Net_2,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=3,padding=1),nn.AvgPool2d(kernel_size=3,padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(64,32,kernel_size=3,padding=1),nn.MaxPool2d(kernel_size=3,padding=1))
        self.layer3 = nn.Sequential(nn.Conv2d(32,16,kernel_size=3,padding=1),nn.MaxPool2d(kernel_size=3,padding=1))
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        out = self.flatten(x)
        return out

class Flower_Net_3(nn.Module):
    def __init__(self):
        super(Flower_Net_3,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,32,kernel_size=3,padding=1),nn.AvgPool2d(kernel_size=3,padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(32,8,kernel_size=3,padding=1),nn.MaxPool2d(kernel_size=3,padding=1))
        self.layer3 = nn.Sequential(nn.Conv2d(8,8,kernel_size=3,padding=1),nn.MaxPool2d(kernel_size=3,padding=1))
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        out = self.flatten(x)
        return out

class ensemble_Net(nn.Module):
    
    def __init__(self):
        super(ensemble_Net,self).__init__()
        f1 = Flower_Net_1()
        f2 = Flower_Net_2()
        f3 = Flower_Net_3()
        self.e1 = f1
        self.e2 = f2
        self.e3 = f3
        self.avgpool = nn.AvgPool1d(kernel_size=1)
        self.fc1 = nn.Linear(10232,3000)
        self.fc2 = nn.Linear(3000,5)
    
    def forward(self,x):
        o1 = self.e1(x)
    
        o2 = self.e2(x)
        o3 = self.e3(x)
        x = torch.cat((o1,o2,o3),dim=1)
        #print(x.size())
        x = self.fc1(x)
        out = self.fc2(x)
        
        return out
    
        
model = ensemble_Net()
model

import torch
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)
#model=vgg
model.to(device)
from torch.optim import Adam,SGD
criterion = nn.CrossEntropyLoss()
optim = Adam(model.parameters(),lr=1e-5,weight_decay=1e-5)



### Training Loop ####
import torch.nn.functional as F 
n_epochs = 8

for epoch in range(n_epochs):
    train_loss = 0
    val_loss = 0
    acc = 0.0
    print("Training....")
    model.train()
    for batch_num,(batch,labels) in enumerate(train_loader):
        inp,target = batch.to(device),labels.to(device)
        optim.zero_grad()
        output = model.forward(inp)
        
        op = F.softmax(output,dim=1)
        
        final_op = torch.argmax(op,dim=1)
        
        acc += torch.sum(final_op==target).item()/len(target)
        loss = criterion(output,target)
        
        loss.backward()
        optim.step()
        
        train_loss+=(loss.item()/len(batch))
        if batch_num%50 ==0 and batch_num!=0:
            print("TARGET: ",target)
            print("OUTPUT: ",final_op)
            print("Accuracy after ",batch_num,"steps: ",acc/batch_num)
        
    
    acc = acc/len(train_loader)
    print("Epoch: ",epoch,"Loss: ",train_loss," Accuracy: ",acc)
    
    
    eval_acc = 0

    model.eval()
    print("Validating.....")
    for batch in valid_loader:
        inp,target = batch[0].to(device),batch[1].to(device)
        op = F.softmax(model.forward(inp))
        final_op = torch.argmax(op,dim=1)
       
    
        eval_acc += np.sum(final_op.detach().cpu().numpy()==target.detach().cpu().numpy())/len(target)
        
    print("Validation accuracy: ",eval_acc/len(valid_loader))
    #print("FOP",final_op)
    #print("TARGET",target)
    
    
    

            

torch.save(model,"ensemble.pt")
### Single Image Results #####
batch,label = next(iter(valid_loader))
img = batch[0]
label = label[0]
plt.imshow(img.permute(1,2,0))
img = torch.reshape(img,(1,3,224,224))
img = img.to(device)
with torch.no_grad():
    op = model.forward(img)
    out = torch.argmax(F.softmax(op))
actual = flower_dataset.classes[label]
pred = flower_dataset.classes[out]
print("Out",out.item())
print("Predicted",pred)
print("Actual",actual)
import torchvision
from torchvision.models import vgg16
vgg = torchvision.models.vgg16(pretrained=True)
import torch.nn as nn
vgg.classifier[6] = nn.Linear(in_features=4096,out_features=5,bias=True)
from torch.optim import Adam
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vgg.to(device)
from torch.optim import Adam,SGD
criterion = nn.CrossEntropyLoss()
optim = Adam(vgg.parameters(),lr=1e-5,weight_decay=1e-5)

### Training Loop ####
import torch.nn.functional as F 
n_epochs = 8
max_acc=0
for epoch in range(n_epochs):
    train_loss = 0
    val_loss = 0
    acc = 0.0
    print("Training....")
    vgg.train()
    for batch_num,(batch,labels) in enumerate(train_loader):
        inp,target = batch.to(device),labels.to(device)
        optim.zero_grad()
        output = vgg.forward(inp)
        
        op = F.softmax(output,dim=1)
        
        final_op = torch.argmax(op,dim=1)
        
        acc += torch.sum(final_op==target).item()/len(target)
        loss = criterion(output,target)
        
        loss.backward()
        optim.step()
        
        train_loss+=(loss.item()/len(batch))
        if batch_num%50 ==0 and batch_num!=0:
            print("TARGET: ",target)
            print("OUTPUT: ",final_op)
            print("Accuracy after ",batch_num,"steps: ",acc/batch_num)
        
    
    acc = acc/len(train_loader)
    print("Epoch: ",epoch,"Loss: ",train_loss," Accuracy: ",acc)
    
    
    eval_acc = 0

    vgg.eval()
    print("Validating.....")
    for batch in valid_loader:
        inp,target = batch[0].to(device),batch[1].to(device)
        op = F.softmax(vgg.forward(inp))
        final_op = torch.argmax(op,dim=1)
           
    
        eval_acc += np.sum(final_op.detach().cpu().numpy()==target.detach().cpu().numpy())/len(target)
        
    print("Validation accuracy: ",eval_acc/len(valid_loader))
    if eval_acc>max_acc:
        max_acc = eval_acc
        torch.save(vgg,"vgg.pt")
    #print("FOP",final_op)
    #print("TARGET",target)
    
    
    

            


