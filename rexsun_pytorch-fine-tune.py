import numpy as np

import pandas as pd

import os

import torch

import torchvision

import torch.nn as nn

import torch.functional as F

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader,Dataset

from torchvision import datasets,models,transforms

from PIL import Image

from sklearn.model_selection import StratifiedShuffleSplit

torch.version
os.listdir("../input")
all_labels_df=pd.read_csv("../input/labels.csv")

all_labels_df.head()
breeds=all_labels_df.breed.unique()

breed2idx=dict((breed,idx) for idx,breed in enumerate(breeds))

idx2breed=dict((idx,breed) for idx,breed in enumerate(breeds))

print(len(breeds))   # 这个数据有120个类别
all_labels_df["label_idx"]=[breed2idx[b] for b in all_labels_df.breed]

all_labels_df.head()
class DogDataset(Dataset):

    def __init__(self,label_df,img_path,transform=None):

        self.label_df=label_df

        self.img_path=img_path

        self.transform=transform

        

    def __len__(self):

        """放回数据集的长度"""

        return self.label_df.shape[0]

    

    def __getitem__(self,idx):

        """读取图片和标签"""

        label=self.label_df.label_idx[idx]

        id_img=self.label_df.id[idx]

        img_P=os.path.join(self.img_path,id_img)+".jpg"

        img=Image.open(img_P)

        

        if self.transform:

            img=self.transform(img)

            

        return img,label
IMG_SIZE=224   # resnet50的输入是224的，所以需要将图片统一大小

BATCH_SIZE=256   # 每个批次输入的图片的数量

IMG_MEAN=[0.485,0.456,0.406]

IMG_STD=[0.229,0.224,0.225]

CUDA=torch.cuda.is_available()

DEVICE=torch.device("cuda" if CUDA else "cpu")
train_transform=transforms.Compose([

    transforms.Resize(IMG_SIZE),   # 改变图片大小

    transforms.RandomResizedCrop(IMG_SIZE),  # 首先随机裁减，然后在转换成规定图片大小

    transforms.RandomHorizontalFlip(),  # 以0.5的概率随机水平翻转

    transforms.RandomRotation(30),  # 旋转

    transforms.ToTensor(),  # 转换成张量

    transforms.Normalize(IMG_MEAN,IMG_STD)  # 标准化

])



val_transform=transforms.Compose([

    transforms.Resize(IMG_SIZE),

    transforms.CenterCrop(IMG_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(IMG_MEAN,IMG_STD)

])
# 使用分层抽样切分训练集和验证集

dataset_name=["train","valid"]

stra_splt=StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=0)

train_index,val_index=next(iter(stra_splt.split(all_labels_df.id,all_labels_df.breed)))

train_df=all_labels_df.iloc[train_index,:].reset_index()

val_df=all_labels_df.iloc[val_index,:].reset_index()

print(len(train_df))

print(len(val_df))
image_tarnsforms={"train":train_transform,"valid":val_transform}



train_dataset=DogDataset(train_df,"../input/train",transform=image_tarnsforms["train"])

valid_dataset=DogDataset(val_df,"../input/train",transform=image_tarnsforms["valid"])

image_dataset={"train":train_dataset,"valid":valid_dataset}



image_dataloader={x:DataLoader(image_dataset[x],batch_size=BATCH_SIZE,shuffle=True,num_workers=0) for x in dataset_name}

datasize={x:len(image_dataset[x]) for x in dataset_name}
model_ft=models.resnet50(pretrained=True) # 自动下载官方的预训练模型

# 将所有的层都先冻结

for param in model_ft.parameters():

    param.requires_grad=False



# 打印全连接层的信息

print(model_ft.fc)

num_fc_ftr=model_ft.fc.in_features  # 获取全连接层的输入

model_ft.fc=nn.Linear(num_fc_ftr,len(breeds))  # 定义一个新的全连接层

model_ft=model_ft.to(DEVICE)  # 将网络放到设备中

print(model_ft)  # 最后打印一下模型
criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam([{"params":model_ft.fc.parameters()}],lr=0.001)  # 指定新加的fc层的学习率
for data in image_dataloader["train"].dataset:

    x,y=data

    plt.imshow(x)

    plt.title(y)

    break
def train(model,device,train_loader,epoch):

    model.train()

    for batch_idx,data in enumerate(train_loader):

        x,y=data

        x=x.to(device)

        y=y.to(device)

        optimizer.zero_grad()

        y_hat=model(x)

        loss=criterion(y_hat,y)

        loss.backward()

        optimizer.step()

    print ('Train Epoch: {}\t Loss: {:.6f}'.format(epoch,loss.item()))
def test(model,device,test_loader):

    model.eval()

    test_loss=0

    correct=0

    with torch.no_grad():

        for i,data in enumerate(test_loader):

            x,y=data

            x=x.to(device)

            y=y.to(device)

            optimizer.zero_grad()

            y_hat=model(x)

            test_loss+=criterion(y_hat,y).item()

            pred=y_hat.max(1,keepdim=True)[1]

            correct+=pred.eq(y.view_as(pred)).sum().item()

    test_loss/=len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(valid_dataset),

        100. * correct / len(valid_dataset)))
for epoch in range(1,9):

    %time train(model_ft,DEVICE,image_dataloader["train"],epoch)

    test(model_ft,DEVICE,image_dataloader["valid"])
# hook

in_list=[]  # 存放所有的输出

def hook(module,input,output):

    for i in range(input[0].size(0)):

        in_list.append(input[0][i].cpu().numpy())
model_ft.avgpool.register_forward_hook(hook)
%%time

with torch.no_grad():

    for batch_idx,data in enumerate(image_dataloader["train"]):

        x,y=data

        x=x.to(DEVICE)

        y=y.to(DEVICE)

        y_hat=model_ft(x)
features=np.array(in_list)
np.save("/kaggle/working/features",features)
features.shape