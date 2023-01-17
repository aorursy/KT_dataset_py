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
        array = array.reshape((512, 512, 1))
        return np.stack([array, array, array], axis=2).reshape((512, 512, 3))
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
            img=img.reshape((512,512,1)).astype('float')
            
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
            img=img.reshape((512,512,1)).astype('float')
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
split = int(np.floor(0.2 * len(train_ds)))
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
        self.model=torchvision.models.mobilenet_v2(pretrained=True)
        for p in self.model.parameters():
            p.requires_grad=False
        self.last_Channel=1000
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.nepe=nn.Sequential(
        nn.Dropout(0.2),
            nn.Linear(self.last_Channel,1),
            nn.Sigmoid()
        )
        
        self.rlrg1=nn.Sequential(
        nn.Dropout(0.2),
            nn.Linear(self.last_Channel,1),
            nn.Sigmoid()
        )
        
        self.rlrl1=nn.Sequential(
        nn.Dropout(0.2),
            nn.Linear(self.last_Channel,1),
            nn.Sigmoid()
        )
        
        self.lspe =nn.Sequential(
        nn.Dropout(0.2),
            nn.Linear(self.last_Channel,1),
            nn.Sigmoid()
        )
        
        self.cpe =nn.Sequential(
        nn.Dropout(0.2),
            nn.Linear(self.last_Channel,1),
            nn.Sigmoid()
        )
        
        self.rspe =nn.Sequential(
        nn.Dropout(0.2),
            nn.Linear(self.last_Channel,1),
            nn.Sigmoid()
        )
        
        self.aacpe=nn.Sequential(
        nn.Dropout(0.2),
            nn.Linear(self.last_Channel,1),
            nn.Sigmoid()
        )
        
        self.cnpe =nn.Sequential(
        nn.Dropout(0.2),
            nn.Linear(self.last_Channel,1),
            nn.Sigmoid()
        )
        
        self.indt =nn.Sequential(
        nn.Dropout(0.2),
            nn.Linear(self.last_Channel,1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.model(x)

        return {
            'negative_exam_for_pe': self.nepe(x),
            'rv_lv_ratio_gte_1': self.rlrg1(x),
            'rv_lv_ratio_lt_1': self.rlrl1(x),
            'leftsided_pe': self.lspe(x),
            'chronic_pe': self.cpe(x),
            'rightsided_pe': self.rspe(x),
            'acute_and_chronic_pe': self.aacpe(x),
            'central_pe': self.cnpe(x),
            'indeterminate': self.indt(x)
        }
    def get_loss(self, net_output, ground_truth):
        negative_exam_for_pe = F.binary_cross_entropy(net_output['negative_exam_for_pe'].float(), ground_truth[:,0])
        rv_lv_ratio_gte_1= F.binary_cross_entropy(net_output['rv_lv_ratio_gte_1'].float(), ground_truth[:,1])
        rv_lv_ratio_lt_1 = F.binary_cross_entropy(net_output['rv_lv_ratio_lt_1'].float(), ground_truth[:,2])
        leftsided_pe = F.binary_cross_entropy(net_output['leftsided_pe'].float(), ground_truth[:,3])
        chronic_pe= F.binary_cross_entropy(net_output['chronic_pe'].float(), ground_truth[:,4])
        rightsided_pe = F.binary_cross_entropy(net_output['rightsided_pe'].float(), ground_truth[:,5])
        acute_and_chronic_pe = F.binary_cross_entropy(net_output['acute_and_chronic_pe'].float(), ground_truth[:,6])
        central_pe= F.binary_cross_entropy(net_output['central_pe'].float(), ground_truth[:,7])
        indeterminate = F.binary_cross_entropy(net_output['indeterminate'].float(), ground_truth[:,8])
        loss = negative_exam_for_pe+rv_lv_ratio_gte_1+rv_lv_ratio_lt_1+leftsided_pe+chronic_pe+rightsided_pe+acute_and_chronic_pe+central_pe+indeterminate
        
        return loss, {'negative_exam_for_pe':negative_exam_for_pe,
            'rv_lv_ratio_gte_1': rv_lv_ratio_gte_1,
            'rv_lv_ratio_lt_1': rv_lv_ratio_lt_1,
            'leftsided_pe': leftsided_pe,
            'chronic_pe': chronic_pe,
            'rightsided_pe': rightsided_pe,
            'acute_and_chronic_pe': acute_and_chronic_pe,
            'central_pe': central_pe,
            'indeterminate': indeterminate}



gc.collect()
device='cuda' if torch.cuda.is_available() else 'cpu'
model=train_model()
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters())

epochs=1

for epoch in range(epochs):
    total_loss=0
    model.train()
    for i,(img,y) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        img,y=img.float().to(device),y.float().to(device)
        outputs=model(img)
        loss_train, losses_train = model.get_loss(outputs,y)
        total_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
    print("*"*80)
    print("")
    print("Train")
    print("")
    print(losses_train/len(train_loader))
    print("")
    model.eval()
    for i,(img,y) in enumerate(test_loader):
        
        img,y=img.float().to(device),y.to(device)
        outputs=model(img)
        loss_train, losses_train = model.get_loss(outputs, y.float())
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
    
