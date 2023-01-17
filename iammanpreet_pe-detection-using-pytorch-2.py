!conda install -c conda-forge gdcm -y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import glob
import gdcm
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import pydicom as dcm
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
warnings.filterwarnings('ignore')
import gc
from skimage.transform import resize
path="../input/rsna-str-pulmonary-embolism-detection/"
train=pd.read_csv(path+"train.csv")
test=pd.read_csv(path+"test.csv")
sub=pd.read_csv(path+"sample_submission.csv")
print("shape of train dataframe : {}".format(train.shape))
print("shape of test dataframe : {}".format(test.shape))
print("shape of submission dataframe : {}".format(sub.shape))
train.head()
fig, ax = plt.subplots(figsize=(8, 8))
img=dcm.dcmread("../input/rsna-str-pulmonary-embolism-detection/train/4833c9b6a5d0/57e3e3c5f910/f4fdc88f2ace.dcm").pixel_array
ax.imshow(img)
print(img.shape)
class bw_to_rgb():
    def __call__(self,array):
        array = array.reshape((128, 128, 1))
        return np.stack([array, array, array], axis=2).reshape((128, 128, 3))
    def __repr__(self):
        return self.__class__.__name__ + '()'

class getdata(Dataset):
    def __init__(self,df,mode="train",transform=None):
        self.mode=mode
        self.transform=transform
        self.df=df
        self.path="../input/rsna-str-pulmonary-embolism-detection/"
    def __getitem__(self,idx):
        fnames = self.df[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']]
        if self.mode=="train":
            stuid=fnames.loc[idx].values[0]
            siuid=fnames.loc[idx].values[1]
            souid=fnames.loc[idx].values[2]
            img=dcm.dcmread(self.path+"train/"+stuid+"/"+siuid+"/"+souid+".dcm").pixel_array
            img = resize(img, (128,128), anti_aliasing=True).astype('float')
            #img=img.reshape((512,512,1)).astype('float')
            
            y=self.df[['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                     'leftsided_pe', 'chronic_pe', 'rightsided_pe',
                     'acute_and_chronic_pe', 'central_pe', 'indeterminate']].loc[idx].values
            if self.transform:
                img=transform(img)
            return img,y
        else:
            stuid=fnames.loc[idx].values[0]
            siuid=fnames.loc[idx].values[1]
            souid=fnames.loc[idx].values[2]
            img=dcm.dcmread(self.path+"test/"+stuid+"/"+siuid+"/"+souid+".dcm").pixel_array
            img = resize(img, (128,128), anti_aliasing=True).astype('float')
            #img=img.reshape((512,512,1)).astype('float')
            if self.transform:
                img=transform(img)
            return img
    def __len__(self):
        return len(self.df)
            
# img=next(iter(getdata(test,mode="test",transform=transform)))
transform=transforms.Compose([
    bw_to_rgb(),
    transforms.ToTensor()
])

train_ds=getdata(train,transform=transform)
indices=list(range(len(train_ds)))
split = int(np.floor(0.35 * len(train_ds)))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_ds, batch_size=64,
                                                sampler=valid_sampler)
len(train_loader),len(train_ds)
for a,(i,y) in enumerate(train_loader):
    print(i.shape)
    print(y.shape)
    break
y[:,0]
# model=model.cpu()
# i=i.cpu()
# model(i)
class train_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1=nn.Conv2d(3,32,kernel_size=(3,3),padding=1,stride=1)
        self.cn2=nn.Conv2d(32,60,kernel_size=(3,3),padding=1,stride=1)
        self.cn3=nn.Conv2d(60,128,kernel_size=(3,3),padding=1,stride=1)
        self.cn4=nn.Conv2d(128,128,kernel_size=(3,3),padding=1,stride=1)
        self.pool=nn.MaxPool2d(kernel_size=(2,2))
        
        self.l1=nn.Linear(8192,128)
        self.l2=nn.Linear(128,100)
        self.l3=nn.Linear(100,9)
        
        
    def forward(self,x):
        x=self.pool(self.cn1(x))
        x=self.pool(self.cn2(x))
        x=self.pool(self.cn3(x))
        x=self.pool(self.cn4(x))
        x=x.view(x.shape[0],-1)
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=self.l3(x)
        return x
# model=train_model()
# x=model(i.float())
# x.shape

gc.collect()
device='cuda' if torch.cuda.is_available() else 'cpu'
model=train_model()
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters())
loss=nn.BCEWithLogitsLoss()
epochs=1

for epoch in range(epochs):
    total_loss=0
    model.train()
    for i,(img,y) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        img,y=img.float().to(device),y.float().to(device)
        outputs=model(img)
        loss_train= loss(outputs,y)
        total_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
    print("*"*80)
    print("")
    print("Train")
    print("")
    print(total_loss/len(train_loader))
    print("")
    model.eval()
    for i,(img,y) in enumerate(test_loader):
        
        img,y=img.float().to(device),y.to(device)
        outputs=model(img)
        loss_train= loss(outputs, y.float())
        total_loss += loss_train.item()
    print("val loss : ",loss_train)
    print("")
    print("*"*80)
    print("*"*80)
    print("")
    print(epoch+" : "+ total_loss/len(test_loader))
    print("")
    print("*"*80)
    print("*"*80)
    
for i,(img,y) in tqdm(enumerate(train_loader)):
    img,y=img.float().to(device),y.float().to(device)
    outputs=model(img)
    break
img.shape
model
