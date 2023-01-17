import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms,utils
from torchvision.utils import make_grid
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import torchvision
df=pd.read_csv("../input/jovian-pytorch-z2g/Human protein atlas/train.csv")
dic={0: 'Mitochondria',1: 'Nuclear bodies',2: 'Nucleoli',3: 'Golgi apparatus',4: 'Nucleoplasm',5: 'Nucleoli fibrillar center',6: 'Cytosol',7: 'Plasma membrane',8: 'Centrosome',9: 'Nuclear speckles'}
print(dic)
print(df.head())
length=len(df)
def encode_label(df):
    
    labels=[]
    for i in range(length):
        temp=np.zeros(10,dtype="int32")
        df["Image"][i]=str(df["Image"][i])+".png"
        for label in str(df["Label"][i]).strip().split(" "):
            temp[int(label)]=int(1)   
        l=""
        for x in temp:
            l+=str(x)+" "
        df["Label"][i]=l
    return(df)    
            
    
    
df=encode_label(df)
print(df.head())
df.to_csv("new_train.csv")
image,labels=(df["Image"][12],df["Label"][12])
print(image,len(labels))
def decode(out,dic,to_text=False):
    label_to_text=[]
    decode_label_to_text=[]
    decode_label_to_index=[]
    label_to_index=[]
    i=0
    for i in range(len(out)):
        for j in range(len(out[0])):
            #out[j]
            if(out[i][j]==1):
                label_to_text.append([dic[j],j])
                label_to_index.append(j)
        decode_label_to_text.append(label_to_text)
        decode_label_to_index.append(label_to_index)
    if(to_text==True):
        return decode_label_to_text
    else:
        return decode_label_to_index
    
            
labels=decode(torch.tensor([[0,0,0,0,1,0,1,0,0,1]]),dic,True)
labels
class HumanProtienDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.root_dir=root_dir
        self.data=pd.read_csv(csv_file)
        self.transform=transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        image,label=(self.data["Image"][index],self.data["Label"][index])
        #print(image)
        label=label.strip().split(" ")
        label=torch.tensor([float(i) for i in label])
        img=Image.open(self.root_dir+image)
        #img=cv2.resize(img,(224,224))
        if self.transform:
            img=self.transform(img)
        return img,label    
np.random.seed(42)
msk = np.random.rand(len(df)) < 0.9

train_df = df[msk].reset_index()
val_df = df[~msk].reset_index() 
train_df.drop(["index"],axis=1)
val_df.drop(['index'],axis=1)
train_df.to_csv("df_train.csv")
val_df.to_csv("df_val.csv")
print(len(train_df),len(val_df))
train_dir = '../input/jovian-pytorch-z2g/Human protein atlas/train/'
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = transforms.Compose([
    transforms.RandomCrop(512, padding=8, padding_mode='reflect'),
    transforms.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
#     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10),
    transforms.ToTensor(), 
    #transforms.Normalize(*imagenet_stats,inplace=True), 
    transforms.RandomErasing(inplace=True)
])

valid_tfms = transforms.Compose([
    transforms.Resize(256), 
    transforms.ToTensor(), 
#     T.Normalize(*imagenet_stats)
])
train_dataset=HumanProtienDataset("df_train.csv",train_dir,train_tfms)
val_dataset=HumanProtienDataset("df_val.csv",train_dir,valid_tfms)
print(len(train_dataset))
print(len(val_dataset))
def show_sample(img, target, dic,invert=True):
   # print(target,type(target))
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    #print('Labels:', decode([target],dic,True))
show_sample(*train_dataset[0], dic,invert=False)
train_dataset[0][0].shape
batch_size=64
train_dl=DataLoader(train_dataset,batch_size,shuffle=True,num_workers=0,pin_memory=True)
val_dl=DataLoader(val_dataset,batch_size,num_workers=0,pin_memory=True)
def show_batch(dl, invert=True):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
        break
show_batch(train_dl)
def F_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)
class Model(nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        out = self(images)                      
        loss = F.binary_cross_entropy(out, targets)      
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        out = self(images)                           # Generate predictions
        loss = F.binary_cross_entropy(out, targets)  # Calculate loss
        score = F_score(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach() }
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))
class block(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,identity_downsample=None):
        super(block,self).__init__()
        self.expansion=4
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=stride,padding=1)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.conv3=nn.Conv2d(out_channel,out_channel*self.expansion,kernel_size=1,stride=1,padding=0)
        self.bn3=nn.BatchNorm2d(out_channel*self.expansion)
        self.relu=nn.ReLU()
        self.downsample=identity_downsample
        self.stride=stride
    def forward(self,x):
        identity=x
        #print(x.shape)
        x=self.conv1(x)
        #print(x.shape)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        #print(x.shape)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.conv3(x)
        #print(x.shape)
        x=self.bn3(x)
        x=self.relu(x)
        
        if self.downsample is not None:
            identity=self.downsample(identity)
        #print(x.shape)
        x=x+identity
        x=self.relu(x)
        return x
class Resnet(Model):
    def __init__(self,block,image_channel,layers,num_classes):
        super(Resnet, self).__init__()
        self.in_channel=64
        self.conv1=nn.Conv2d(image_channel,64,kernel_size=7,stride=2,padding=3)
        self.pool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.layer1=self.res_layer(block,layers[0],out_channel=64,stride=1)
        self.layer2=self.res_layer(block,layers[1],out_channel=128,stride=2)
        self.layer3=self.res_layer(block,layers[2],out_channel=256,stride=2)
        self.layer4=self.res_layer(block,layers[3],out_channel=512,stride=2)
        self.adpavgpool=nn.AdaptiveAvgPool2d((1,1))
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(2048,num_classes)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.pool(x)
        #print("after pool",x.shape)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.adpavgpool(x)
        #print("avg_p00l",x.shape)
        x=self.flatten(x)
        #print("flatten",x.shape)
        x=self.linear1(x)
        out=self.sigmoid(x)
        return out
    
    def res_layer(self,block,num_layer,out_channel,stride):
        layers=[]
        identity=None
        if stride!=1 or self.in_channel!=out_channel*4:
            identity=nn.Sequential(nn.Conv2d(self.in_channel,out_channel*4,kernel_size=1,stride=stride),
                                   nn.BatchNorm2d(out_channel*4))
        
        layers.append(block(self.in_channel,out_channel,stride,identity))
        self.in_channel=out_channel*4
        for i in range(num_layer-1):
            layers.append(block(self.in_channel,out_channel,stride=1))
            
        return nn.Sequential(*layers)    
def ResNet50(img_channel=3, num_classes=10):
    return Resnet(block,img_channel,[3, 4, 6, 3],num_classes)
model=ResNet50()
model_conv=torchvision.models.resnet50(pretrained=True)

torch.save(model_conv.state_dict(),"pretrained.pth")
model.load_state_dict(torch.load("../input/weight/Resnet50_epoch10 (1).pth"),strict=False)
count=0
for parameter in model.parameters():
    #print(count,parameter.data.shape)
    parameter.requires_grad =True
    count+=1
    if(count==200):
        break
for parameter in model.parameters():
    if parameter.requires_grad :
         print(parameter.data.shape)   
    
    
print(count)    
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
device = get_default_device()
device
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device);
def try_batch(dl):
    for images, labels in dl:
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]:', out[0])
        break

try_batch(train_dl)
from tqdm.notebook import tqdm
val_dl = DeviceDataLoader(val_dl, device)
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        name='Resnet50_epoch'+str(epoch)+'.pth'
        torch.save(model.state_dict(),name)
    return history
model = to_device(model, device)
print(evaluate(model, val_dl))

opt_func = torch.optim.SGD
lr = 0.05
history = fit(25, lr, model, train_dl, val_dl, opt_func)
opt_func = torch.optim.SGD
lr = 0.04
history = fit(25, lr, model, train_dl, val_dl, opt_func)
opt_func = torch.optim.SGD
lr = 0.01
history = fit(25, lr, model, train_dl, val_dl, opt_func)
opt_func = torch.optim.SGD
lr = 0.001
history = fit(25, lr, model, train_dl, val_dl, opt_func)
opt_func = torch.optim.SGD
lr = 0.0001
history = fit(10, lr, model, train_dl, val_dl, opt_func)
