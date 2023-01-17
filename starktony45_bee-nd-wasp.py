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
!pip install torchsummary
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2 
from torchsummary import summary
import time
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('/kaggle/input/bee-vs-wasp/kaggle_bee_vs_wasp/labels.csv',index_col =False)
for i in data.index:
    data['path'].iloc[i] = data['path'].iloc[i].replace('\\', '/')
le = LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])
data[:100]
data.info()
valid_img = data[data['is_validation']==1]
type(valid_img.label)
final_valid = data[data['is_final_validation']==1]
type(final_valid.label)
train_img = data[(data['is_validation']==0) & (data['is_final_validation']==0)]
type(train_img)
w = train_img[:4]
dir = '/kaggle/input/bee-vs-wasp/kaggle_bee_vs_wasp'
lis = w['path'].tolist()
plt.figure(figsize=(10,10))
for iterator,filename in enumerate(lis):
    images = Image.open(os.path.join(dir,filename))
    plt.subplot(4,2,iterator+1)
    plt.imshow(images,cmap=plt.cm.bone)
    
plt.tight_layout()
valid_img.label = valid_img.label.astype(np.int64)
final_valid.label = final_valid.label.astype(np.int64)
import torch
from torchvision import datasets,transforms,models
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch.optim import Adam,SGD

train_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.225,0.224])])

valid_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.225,0.224])])
class BeeDataset(Dataset):
    def __init__(self, df:pd.DataFrame, imgdir:str, train:bool,
                 transforms=None):
        self.df = df
        self.imgdir = imgdir
        self.train = train
        self.transforms = transforms
    
    def __getitem__(self, index):
        im_path = os.path.join(self.imgdir, self.df.iloc[index]["path"])
        x = cv2.imread(im_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (224, 224))

        if self.transforms:
            x = self.transforms(x)
        
        if self.train:
            y = self.df.iloc[index]["label"]
            return x, y
        else:
            return x
    
    def __len__(self):
        return len(self.df)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = models.resnet18(pretrained = True)
        self.model.fc = nn.Linear(512,4)
    def forward(self,x):
        output = self.model(x)
        return output
type(valid_img)
#train_img.label = train_img.label.astype(np.int64)

train_dataset = BeeDataset(df =train_img,imgdir = '/kaggle/input/bee-vs-wasp/kaggle_bee_vs_wasp',train=True,
                          transforms = train_transform)
valid_dataset = BeeDataset(df =valid_img,imgdir = '../input/bee-vs-wasp/kaggle_bee_vs_wasp',train=True,
                          transforms = valid_transform)
test_dataset = BeeDataset(df =final_valid,imgdir = '../input/bee-vs-wasp/kaggle_bee_vs_wasp',train=True,
                          transforms = valid_transform)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
print(device)
criterion = nn.CrossEntropyLoss()

arch = Net()
arch.to(device)
    
optim = torch.optim.SGD(arch.parameters(),lr = 1e-3,momentum =0.9)
train_loader = DataLoader(dataset = train_dataset,shuffle = True,batch_size=32,num_workers = 4)
valid_loader = DataLoader(dataset = valid_dataset,shuffle = True,batch_size=32,num_workers = 4)
test_loader  = DataLoader(dataset = test_dataset, shuffle = True,batch_size=32,num_workers = 4)

summary(model = arch, input_size = (3, 224, 224),batch_size =32)
def train_model(model,optimizer,n_epochs,criterion):
    start_time = time.time()
    for epoch in range(1,n_epochs-1):
        epoch_time = time.time()
        epoch_loss = 0
        correct = 0
        total = 0
        print( "Epoch {}/{}".format(epoch,n_epochs))
        
        model.train()
        
        for inputs,labels in train_loader:
            inputs = inputs.to(device)
            
            labels  = (labels).to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            epoch_loss +=loss.item()
            _,pred =torch.max(output,1)
            correct += (pred.cpu()==labels.cpu()).sum().item()
            total +=labels.shape[0]
            
        acc = correct/total
        
        model.eval()
        a= 0
        pred_val = 0
        corr = 0
        tot = 0
        
        with torch.no_grad():
            for val_inp,val_label in valid_loader:
                val_inp = val_inp.to(device)
                val_label = val_label.to(device)
                out_val = model(val_inp)
                loss = criterion(out_val,val_label)
                a += loss.item()
                _,pred_val = torch.max(out_val,1)
                corr += (pred_val.cpu()==val_label.cpu()).sum().item()
                tot = val_label.shape[0]
            acc_val = corr/tot
        epoch_time2 = time.time()
        print("Duration : {:.4f},Train Loss :{:.4f},Train Acc :{:.4f}, Valid Loss:{:.4f},Valid acc :{:.4f}".format(
        epoch_time2-epoch_time,epoch_loss/len(labels),acc,a/len(val_label),acc_val))
    end_time= time.time()
    print("Total time :{:.0f}s".format(end_time - start_time))
            
def eval_model(model):
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for images,label in test_loader:
            images = images.to(device)
            label = label.to(device)
            output = model(images)
            _, pred = torch.max(output,1)
            correct += (pred == label).sum().item()
            total += label.shape[0]
    print("The accuracy in Test dataset is %d %%" %(100*correct/total))
            
    
train_model(model=arch, optimizer=optim, n_epochs=20, criterion=criterion)
eval_model(arch)


