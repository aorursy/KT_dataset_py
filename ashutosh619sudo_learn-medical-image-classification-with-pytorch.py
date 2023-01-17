# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install efficientnet-pytorch
train = pd.read_csv("../input/jpeg-melanoma-256x256/train.csv")
test = pd.read_csv("../input/jpeg-melanoma-256x256/test.csv")
train.head()
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms as T
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
from sklearn.metrics import accuracy_score, roc_auc_score
from efficientnet_pytorch import EfficientNet
from torchvision import models
import pdb
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import matplotlib.pyplot as plt

import pickle 
def get_augmentations(p=0.5):
    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    train_tfms = A.Compose([
        A.Cutout(p=p),
        A.RandomRotate90(p=p),
        A.Flip(p=p),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=50,
                val_shift_limit=50)
        ], p=p),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=p),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=p),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=p),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=p), 
        ToTensor(normalize=imagenet_stats)
        ])
    
    test_tfms = A.Compose([
        ToTensor(normalize=imagenet_stats)
        ])
    return train_tfms, test_tfms
def get_train_val_split(df):
    df = df[df.tfrecord != -1].reset_index(drop=True)
    train_tf_records = list(range(len(df.tfrecord.unique())))[:12]
    split_cond = df.tfrecord.apply(lambda x: x in train_tf_records)
    train_df = df[split_cond].reset_index()
    valid_df = df[~split_cond].reset_index()
    return train_df,valid_df
class MelanomaDataset(Dataset):
    def __init__(self,df,img_path,transform=None,is_test=False):
        self.df = df
        self.img_path = img_path
        self.transform = transform
        self.is_test = is_test
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        img_path = f"{self.img_path}/{self.df.iloc[index]['image_name']}.jpg"
        img = Image.open(img_path)
        
        if self.transform:
            img = self.transform(**{"image": np.array(img)})["image"]
        if self.is_test:
            return img
        target = self.df.iloc[index]["target"]
        return img,torch.tensor([target],dtype=torch.float32)
class MelanomaEfficientNet(nn.Module):
    def __init__(self,model_name="efficientnet-b0",pool_type=F.adaptive_avg_pool2d):
        super().__init__()
        self.pool_type = pool_type
        self.model = EfficientNet.from_pretrained(model_name)
        in_features = 2560  #number of features obtained after image passing through different layers(check in efficientnet library)
        self.classifier = nn.Linear(in_features,1)
    
    def forward(self,x):
        features = self.pool_type(self.model.extract_features(x),1) # extract_features extract the features from the image.Vector length of 2560.
        features = features.view(x.size(0),-1)
        return self.classifier(features)
path = "../input/jpeg-melanoma-256x256"
def data(train_df,valid_df,train_tfms,test_tfms,bs):
    train_ds = MelanomaDataset(df=train_df,img_path=path+"/train",transform=train_tfms)
    valid_ds = MelanomaDataset(df=valid_df,img_path=path+'/train',transform=test_tfms)
    train_dl = DataLoader(dataset=train_ds,batch_size=bs,shuffle=True,num_workers=4)
    valid_dl = DataLoader(dataset=valid_ds,batch_size=bs*2,shuffle=False,num_workers=4)
    return train_dl,valid_dl
X_train,X_val = get_train_val_split(train)
X_train.shape,X_val.shape
train_tfms,test_tfms = get_augmentations(p=0.5)
train_dl,val_dl = data(X_train,X_val,train_tfms,test_tfms,bs=16)
model = MelanomaEfficientNet(model_name="efficientnet-b7")
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.01)
device = torch.device("cuda")
model.to(device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dl)*10)
mb = master_bar(range(10))
mb.write(['epoch','train_loss','valid_loss','val_roc'],table=True)
loss_fn=F.binary_cross_entropy_with_logits
val_rocs = []

for epoch in mb:
    
    train_loss,val_loss = 0.0,0.0
    
    val_preds = np.zeros((len(val_dl.dataset),1))
    val_targs = np.zeros((len(val_dl.dataset),1))
    
    model.train()
    
    for xb,yb in progress_bar(train_dl,parent=mb):
        xb,yb=xb.to(device), yb.to(device)
        
        out = model(xb)
        optimizer.zero_grad()
        
        loss = loss_fn(out,yb)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
    train_loss /= mb.child.total
    print(f"Epoch {epoch}: Training loss: {train_loss}")
    
    model.eval()
    
    with torch.no_grad():
        for i,(xb,yb) in enumerate(progress_bar(val_dl,parent=mb)):
            xb,yb= xb.to(device),yb.to(device)
            
            out = model(xb)
            
            loss =loss_fn(out,yb)
            out = torch.sigmoid(out)
            
            val_loss += loss.item()
            bs = xb.shape[0]
            
            val_preds[i*bs:i*bs+bs] = out.cpu().numpy()
            val_targs[i*bs:i*bs+bs] = yb.cpu().numpy()
            
    val_loss /= mb.child.total
    val_roc = roc_auc_score(val_targs.reshape(-1),val_preds.reshape(-1))
    val_rocs.append(val_roc)
    print(f"Epoch {epoch}: Validation loss: {val_loss}")

    mb.write([epoch,f'{train_loss:.6f}',f'{val_loss:.6f}',f'{val_roc:.6f}'],table=True)
imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
test_tfms = A.Compose([
    A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=50,
                val_shift_limit=50)
        ], p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.5),
    ToTensor(normalize=imagenet_stats)
    ])
test_ds = MelanomaDataset(test,img_path="../input/jpeg-melanoma-256x256/test",is_test=True,transform=test_tfms)
test_dl = DataLoader(dataset = test_ds,batch_size=16,shuffle=False,num_workers=4)
def get_preds(model,device=None,tta=3):
    device=torch.device("cuda")
    model.to(device)
    preds = np.zeros(len(test_ds))
    for tta_id in range(tta):
        test_preds = []
        with torch.no_grad():
            for xb in test_dl:
                xb = xb.to(device)
                out = model(xb)
                out = torch.sigmoid(out)
                test_preds.extend(out.cpu().numpy())
            preds += np.array(test_preds).reshape(-1)
        print(f'TTA {tta_id}')
    preds /= tta
    return preds
preds = get_preds(model,tta=25)  
subm = pd.read_csv("../input/jpeg-melanoma-256x256/sample_submission.csv")
subm.target = preds
subm.to_csv('submission.csv',index=False)