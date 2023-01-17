class CFG:
    debug=False
    #height=256
    #width=256
    lr=1e-4
    batch_size=16
    epochs=5
    seed=777
    target_size=1
    target_col = "label"
    n_fold=4
SIZE = 512

import sys

import gc
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, Counter

import skimage.io
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import scipy as sp

import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,GroupKFold

from functools import partial
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip,RandomGamma, RandomRotate90,GaussNoise,RGBShift,GaussianBlur
from albumentations.pytorch import ToTensorV2
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=42)

!pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b0')
print(model)
!pip install geffnet
import geffnet
model = geffnet.create_model('efficientnet_b0', pretrained=True)
model.classifier=nn.Identity()
a =torch.randn((10,3,512,512))
print(model(a.float()).size())
print(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.current_device())
#データセット作成
import glob
path_list=glob.glob("/kaggle/input/glaucomadataset/Glaucoma/*")
label_list = np.ones(len(path_list))
n_path_list=glob.glob("/kaggle/input/glaucomadataset/Non Glaucoma/*")
n_label_list = np.zeros(len(n_path_list))
path_list.extend(n_path_list)
labels = np.concatenate([label_list, n_label_list])
print(len(path_list),labels.shape)
df = pd.DataFrame(columns =["file","label"])
df["file"] = path_list
df["label"] = labels

df = df.sample(frac=1)
df.head()
df.tail()
df[df["label"]==1]["file"].values[0]
china = pd.read_csv("/kaggle/input/panda-efnetb2-180-weight/china_gla.csv")
china["file"] = ["/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/Training Images/{}".format(china["filename"].values[i]) for i in range(len(china))]
china["label"] = china["Gla"]
china.head()
##concat china_data

china_ = china.drop(["Unnamed: 0","Patient Age","ID","Patient Sex"],axis=1)
china_1 = china_.head(300)
china_0 = china_.tail(300)
print(china_.head())
china_0["from_china"]=1
china_1["from_china"]=1
#df["from_china"]=0
cat_df = pd.concat([china_1,china_0])
print(cat_df.shape)
cat_df.head()

import matplotlib.pyplot as plt
import cv2
%matplotlib inline
idx=0
image = cv2.imread(cat_df['file'].values[idx])
plt.imshow(image)
plt.show()
import math
def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width
def crop_object(img, thresh=10, maxval=200, square=False):
    """
    Source: https://stackoverflow.com/questions/49577973/how-to-crop-the-biggest-object-in-image-with-python-opencv
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# convert to grayscale
    #plt.imshow(gray,cmap="gray")
    #plt.show()#普通に白黒のがみえる
    # threshold to get just the signature (INVERTED)
    retval, thresh_gray = cv2.threshold(gray, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY)
    #plt.imshow(thresh_gray,cmap="gray")
    #plt.show()
    contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #https://qiita.com/anyamaru/items/fd3d894966a98098376c
    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx#(0,0,0,0)なのはcontoursに何も入ってないから
    crop = img[y:y+h, x:x+w]
    if square:
        pad_width = get_pad_width(crop, max(crop.shape))
        crop = np.pad(crop, pad_width=pad_width, mode='constant', constant_values=255)
    return crop

croped= crop_object(image)
plt.imshow(croped)
plt.show()
class TrainDataset(Dataset):
    def __init__(self, df,crop=True,transform1=None, transform2=None):
        self.df = df
        self.crop =crop
        self.transform = transform1
        self.transform_ = transform2
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['file'].values[idx]
        image = skimage.io.MultiImage(file_path)[0]
        if self.crop:
            image = crop_object(image)
        image = cv2.resize(image,(SIZE,SIZE))
        label_ = self.df["label"].values[idx]
        if self.transform:
            image = self.transform(image=image)['image']
        if self.transform_:
            image = self.transform_(image=image)['image']

        
            
        label = torch.tensor(label_).long()
        #print(label_,type(label_),label,label.size())
        
        return image, label
##train_test_split
from sklearn.model_selection import train_test_split
train, test = train_test_split(cat_df, test_size=0.3,stratify = cat_df["label"], random_state=2020)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
##train_valid_split
if CFG.debug:
    folds = train.sample(n=200, random_state=CFG.seed).reset_index(drop=True).copy()
else:
    folds = train.copy()
train_labels = folds["label"].values
kf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for fold, (train_index, val_index) in enumerate(kf.split(folds.values, train_labels)):
    print("num_train,val",len(train_index),len(val_index),len(val_index)+len(train_index))
    folds.loc[val_index, 'fold'] = int(fold)

folds['fold'] = folds['fold'].astype(int)
folds.to_csv('folds.csv', index=None)
folds.head()

import sklearn.metrics as metric

def auc(y,y_hat):
    return metric.roc_auc_score(y,y_hat)

def get_transforms1(*, data):

    #train,valid以外だったら怒る
    
    if data == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            #GaussNoise(p=0.5),
            #RandomRotate90(p=0.5),
            #RandomGamma(p=0.5),
            #RandomAugMix(severity=3, width=3, alpha=1., p=0.5),
            #GaussianBlur(p=0.5),
            #GridMask(num_grid=3, p=0.3),
            #Cutout(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    
    elif data == 'valid':
        return Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

def to_tensor(*args):

        return Compose([
            ToTensorV2()
        ])
base_model = torchvision.models.resnet18(pretrained =True)
base_model.fc = nn.Linear(base_model.fc.in_features, 1)

#画像の確認
import matplotlib.pyplot as plt
%matplotlib inline
dataset = TrainDataset(train.reset_index(drop=True), 
                                 transform1=None,transform2=None)#get_transforms1(data='train')
data_loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=8)
x=0
for img,label in data_loader:
    img = img.detach().numpy()[x]
    print(img.shape,label.detach().numpy()[x])
    plt.imshow(img)
    plt.show()
    break
%%time
for img,label in data_loader:
    img = img.detach().numpy()[0].transpose(1,2,0)
    plt.imshow(img)
    plt.show()
    break
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained =False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        
        
    def forward(self, x):
        x = self.model(x)#ベースのモデルの流れに同じ
        return x
class efenet_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = geffnet.efficientnet_b0(pretrained=True, drop_rate=0.25)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
        
        
    def forward(self, x):
        x = self.model(x)#ベースのモデルの流れに同じ
        return x
class extract_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = geffnet.create_model('efficientnet_b0', pretrained=True)
        self.model.classifier=nn.Identity()
    
        
    def forward(self, x):
        x = self.model(x)#ベースのモデルの流れに同じ
        return x
    
class TrainDataset(Dataset):
    def __init__(self, df, transform1=None, transform2=None):
        self.df = df
        self.transform = transform1
        self.transform_ = transform2
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['file'].values[idx]
        image = skimage.io.MultiImage(file_path)[0]
        image = cv2.resize(image,(SIZE,SIZE))
        label = self.df["label"].values[idx]
        if self.transform:
            image = self.transform(image=image)['image']
        if self.transform_:
            image = self.transform_(image=image)['image']

        
            
        label = torch.tensor(label).float()
        
        return image, label

dataset = TrainDataset(folds.reset_index(drop=True), 
                                 transform1=None,transform2=to_tensor())
loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
##extract vecotr from images
tk0 = tqdm(enumerate(loader), total=len(loader))
embeds =[]
model = extract_Model()
model.to(device)
for i, (images, labels) in tk0:
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        embed = model(images.float())
    embeds.append(embed.cpu().detach().numpy())

embeds_ =np.concatenate(embeds)
print(embeds_.shape,len(folds))
china_dataset = TrainDataset(cat_df.reset_index(drop=True),transform1=None,transform2=to_tensor())
china_loader = DataLoader(china_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
tk1 = tqdm(enumerate(china_loader), total=len(china_loader))
embeds_china =[]
for i, (images, labels) in tk1:
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        embed = model(images.float())
    embeds_china.append(embed.cpu().detach().numpy())
embeds_china_ =np.concatenate(embeds_china)
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
%matplotlib inline
mds = manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=6)
mds_result = mds.fit_transform(embeds_)
#where_from_data = folds["from_china"]
which_Gla = folds["label"]

#plt.scatter(mds_result[:, 0], mds_result[:, 1], c=where_from_data)
#plt.show()
plt.scatter(mds_result[:, 0], mds_result[:, 1], c=which_Gla)
plt.show()
import umap
embedding = umap.UMAP().fit_transform(embeds_)
#plt.scatter(embedding[:, 0], embedding[:, 1], c=where_from_data)
#plt.show()
plt.scatter(embedding[:, 0], embedding[:, 1], c=which_Gla)
plt.show()
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2)
tsne = tsne_model.fit_transform(embeds_)
#plt.scatter(tsne[:, 0], tsne[:, 1], c=where_from_data)
#plt.show()
plt.scatter(tsne[:, 0], tsne[:, 1], c=which_Gla)
plt.show()
which_Gla = cat_df["label"]
embedding_ = umap.UMAP().fit_transform(embeds_china_)
#plt.scatter(embedding[:, 0], embedding[:, 1], c=where_from_data)
#plt.show()
plt.scatter(embedding_[:, 0], embedding_[:, 1], c=which_Gla)
plt.show()
tsne_model = TSNE(n_components=2)
tsne = tsne_model.fit_transform(embeds_china_)
#plt.scatter(tsne[:, 0], tsne[:, 1], c=where_from_data)
#plt.show()
plt.scatter(tsne[:, 0], tsne[:, 1], c=which_Gla)
plt.show()
mds_result_ = mds.fit_transform(embeds_china_)
plt.scatter(mds_result_[:, 0], mds_result_[:, 1], c=which_Gla)
plt.show()

def train_fn(fold):
    print(f"### fold: {fold} ###")

        
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True), 
                                 transform1=None,transform2=to_tensor())#get_transforms1(data='train')
    valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True), 
                                 transform1=None,transform2=to_tensor())#get_transforms1(data='valid')
    
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
    model = efenet_Model()
    #model = Model()
    
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=CFG.lr, amsgrad=False)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True, eps=1e-6)
    #scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)
    
    criterion = nn.BCELoss()#weight = class_weight
    best_score = -100
    best_loss = np.inf
    best_preds = None
    
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()

        model.train()
        """
        if epoch <3:
            for param in model.parameters():
                param.requires_grad = False"""
        avg_loss = 0.

        optimizer.zero_grad()
        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in tk0:
            #if i ==0:
                #print(images.size())

            images = images.to(device)
            labels = labels.to(device)
            
            y_preds = model(images.float())
            y_preds = torch.sigmoid(y_preds.view(-1))
            #if i ==0:
                #print(y_preds.size(),labels.size())#同じ
            loss = criterion(y_preds, labels)
            #loss = criterion(y_preds.view, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss += loss.item() / len(train_loader)
        model.eval()
        avg_val_loss = 0.
        preds = []
        valid_labels = []
        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (images, labels) in tk1:
            
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                y_preds = model(images.float())
                
                y_preds = torch.sigmoid(y_preds.view(-1))
            preds.append(y_preds.to('cpu').numpy())
            valid_labels.append(labels.to('cpu').numpy())

            loss = criterion(y_preds, labels)
            avg_val_loss += loss.item() / len(valid_loader)
        
        #scheduler.step(avg_val_loss)
            
        preds = np.concatenate(preds)
        #print(preds.shape)
        valid_labels = np.concatenate(valid_labels)

        score = auc(valid_labels,preds)

        elapsed = time.time() - start_time
        print(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.6f}  time: {elapsed:.0f}s')
        print(f'  Epoch {epoch+1} - AUC: {score}')
        
        if score>best_score:#aucのスコアが良かったら予測値を更新...best_epochをきめるため
            best_score = score
            best_preds = preds
            print("====",f'  Epoch {epoch+1} - Save Best Score: {best_score:.4f}',"===")
            torch.save(model.state_dict(), f'fold{fold}_resnet18_baseline.pth')#各epochのモデルを保存。。。best_epoch終了時のモデルを推論に使用する？
    
    return best_preds, valid_labels,model
preds = []
valid_labels = []
models =[]
for fold in range(CFG.n_fold):
    _preds, _valid_labels,_model = train_fn(fold)
    preds.append(_preds)
    valid_labels.append(_valid_labels)
    models.append(_model)
##
preds_ = np.concatenate(preds)
valid_labels_ = np.concatenate(valid_labels)

score = auc(valid_labels_,preds_)
import datetime

dt_now = datetime.datetime.now()
print("現在時刻",dt_now)
print("=====AUC(CV)======",score)
train_df = pd.DataFrame()
train_df["predict"] = preds
train_df["label"] = valid_labels
train_df["abs_pred-true"] = np.abs(train_df["predict"]-train_df["label"])
train_df = train_df.sort_values('abs_pred-true', ascending=False)
train_df.head(130)
from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(valid_labels_, preds_)
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%score)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate(Recall)')
plt.grid(True)
plt.show()
class TestDataset(Dataset):
    def __init__(self, df, transform1=None, transform2=None):
        self.df = df
        self.transform = transform1
        self.transform_ = transform2
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['file'].values[idx]
        image = skimage.io.MultiImage(file_path)[0]
        image = cv2.resize(image,(SIZE,SIZE))
        label = self.df["label"].values[idx]
        if self.transform:
            image = self.transform(image=image)['image']
        if self.transform_:
            image = self.transform_(image=image)['image']

        
        return image
    
class baseline_model(nn.Module):

    def __init__(self):
        super().__init__()
        #self.model = torchvision.models.resnet18(pretrained =False)
        #self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model = geffnet.efficientnet_b0(pretrained=False, drop_rate=0.25)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
        
        
    def forward(self, x):
        x = self.model(x)#ベースのモデルの流れに同じ
        return x
def fix_model_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('model.'):
            name = name[6:]  # remove 'model.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

def inference(model, test_loader, device):
    
    model.to(device) 
    
    probs = []

    for i, images in tqdm(enumerate(test_loader), total=len(test_loader)):
            
        images = images.to(device)
            
        with torch.no_grad():
            y_preds = model(images)
            y_preds = torch.sigmoid(y_preds.view(-1))
            
        probs.append(y_preds.to('cpu').numpy())

    probs = np.concatenate(probs)
    
    return probs

def submit():
        print('run inference')
        test_dataset = TestDataset(test, transform1=get_transforms1(data='valid'),transform2=to_tensor())
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
        probs = []
        for fold in range(CFG.n_fold):
            weights_path = "/kaggle/working/fold{}_resnet18_baseline.pth".format(fold)
            model = baseline_model()
            state_dict = torch.load(weights_path,map_location=device)
            model.load_state_dict(state_dict)
            _probs = inference(model, test_loader, device)
            probs.append(_probs)
        probs = np.mean(probs, axis=0)
        return probs
len(test)
test['predict'] = submit()
print(test.head())
score = auc(test['label'].values[:],test['predict'])
print("=====AUC(inner_test)======",score)

def submit():
        print('run inference')
        test_dataset = TestDataset(df, transform1=get_transforms1(data='valid'),transform2=to_tensor())
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
        probs = []
        for fold in range(CFG.n_fold):
            weights_path = "/kaggle/working/fold{}_resnet18_baseline.pth".format(fold)
            model = baseline_model()
            state_dict = torch.load(weights_path,map_location=device)
            model.load_state_dict(state_dict)
            _probs = inference(model, test_loader, device)
            probs.append(_probs)
        probs = np.mean(probs, axis=0)
        return probs

df['predict'] = submit()
print(df.head())
score = auc(df['label'].values[:],df['predict'])
print("=====AUC(inner_test)======",score)
#check test_df
pd.set_option('display.max_rows', 500)
test_df = test
test_df["abs_pred-true"] = np.abs(test_df["predict"]-test_df["label"])
test_df = test_df.sort_values('abs_pred-true', ascending=False)
test_df.head(30)

mistake_file = test_df["file"].values[0]
print(mistake_file)
image = skimage.io.MultiImage(mistake_file)[0]
image = cv2.resize(image,(SIZE,SIZE))
plt.imshow(image)
plt.show()
print("label:1,predict:	0.150904")
mistake_file = test_df["file"].values[1]
print(mistake_file)
image = skimage.io.MultiImage(mistake_file)[0]
image = cv2.resize(image,(SIZE,SIZE))
plt.imshow(image)
plt.show()
print("label:1,predict:0.306142")
china = pd.read_csv("/kaggle/input/panda-efnetb2-180-weight/china_gla.csv")
china.head()
%%time
file_path = china['filename'].values[0]
file_path = "/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images/{}".format(file_path)
image = cv2.imread(file_path)
image = cv2.resize(image,(SIZE,SIZE))
plt.imshow(image)
plt.show()
a = china.head(300)
b = china.tail(300)
china = pd.concat([a,b])
china=china.reset_index()

class TestDataset_china(Dataset):
    def __init__(self, df, transform1=None, transform2=None):
        self.df = df
        self.transform = transform1
        self.transform_ = transform2
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['filename'].values[idx]
        file_path = "/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images/{}".format(file_path)
        image = cv2.imread(file_path)
        image = cv2.resize(image,(SIZE,SIZE))
        label = self.df["Gla"].values[idx]
        if self.transform:
            image = self.transform(image=image)['image']
        if self.transform_:
            image = self.transform_(image=image)['image']

        
        return image
    
weights_path = "/kaggle/working/fold{}_resnet18_baseline.pth".format(0)
model = baseline_model()
state_dict = torch.load(weights_path,map_location=device)
model.load_state_dict(state_dict)
test_dataset = TestDataset_china(china, transform1=get_transforms1(data='valid'),transform2=to_tensor())
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
_probs = inference(model,test_loader, device)

score = auc(china['Gla'].values[:],_probs)
print("=====AUC(china_single_fold)======",score)
china['Gla'].values[:10]
_probs[:10]
test.head(20)
import warnings
warnings.simplefilter('ignore')

class TestDataset(Dataset):
    def __init__(self, df, transform1=None, transform2=None):
        self.df = df
        self.transform = transform1
        self.transform_ = transform2
        
    def __len__(self):
            return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['file'].values[idx]
        image = skimage.io.MultiImage(file_path)[0]
        image = cv2.resize(image,(SIZE,SIZE))
        original = image
        label = self.df["label"].values[idx]
        if self.transform:
            image = self.transform(image=image)['image']
        if self.transform_:
            image = self.transform_(image=image)['image']

        
        return image,original,label
weights_path = "/kaggle/working/fold{}_resnet18_baseline.pth".format(0)
model = baseline_model()
state_dict = torch.load(weights_path,map_location=device)
model.load_state_dict(state_dict)
def getCAM(img,weight_fc,j=0):
    m = torchvision.models.resnet18(pretrained =False)
    m.fc = nn.Linear(m.fc.in_features, 1)
    state_dict = torch.load("/kaggle/working/fold{}_resnet18_baseline.pth".format(0),map_location=device)
    m.load_state_dict(fix_model_state_dict(state_dict))
    m = nn.Sequential(*list(m.children())[:-2])
    m.to(device)
    with torch.no_grad():
        feature_conv = m(img).cpu().detach().numpy()
    bs, nc, h, w = feature_conv.shape
    #print(bs)
    cam = weight_fc.dot(feature_conv[j,:, :, ].reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img

check = train

test_dataset = TestDataset(check, transform1=None,transform2=to_tensor())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model.to(device) 
    
probs = []
fig = plt.figure(figsize=(20, 15))

fc_params = list(model.model.fc.parameters())
weight = np.squeeze(fc_params[0].cpu().data.numpy())
img_size =512

for i, (images,original,label) in tqdm(enumerate(test_loader), total=len(test_loader)):
    images = images.to(device)
    with torch.no_grad():
        y_preds = model(images.float())
        y_preds = torch.sigmoid(y_preds.view(-1))
    if i <10:
        cur_images = images.cpu().permute(0,2,3,1).detach().numpy()
        for j in range(cur_images.shape[0]):
            #print("{0}バッチ目、{1}枚目".format(i,j))
            print('Label:{0}, Predict:{1}'.format(label.view(-1)[j], y_preds[j]))
            ax = fig.add_subplot(100, 200, i+1, xticks=[], yticks=[])
            plt.imshow(cv2.cvtColor(cur_images[j], cv2.COLOR_BGR2RGB))
            #ax.set_title('Label:{0}, Predict:{1}'.format(label.view(-1)[j], y_preds[j]), fontsize=14)
            plt.show()
            heatmap = getCAM(images.float(), weight,j=j)
            ax = fig.add_subplot(100, 200, i+1, xticks=[], yticks=[])
            plt.imshow(cv2.cvtColor(cur_images[j], cv2.COLOR_BGR2RGB))
            plt.imshow(cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_LINEAR), alpha=0.5, cmap='jet')
            plt.show()
            #if j==0:break
        
            
    
    probs.append(y_preds.to('cpu').numpy())

probs = np.concatenate(probs)
print("AUC",auc(check['label'].values[:],probs))

class TrainDataset(Dataset):
    def __init__(self, df, transform1=None, transform2=None):
        self.df = df
        self.transform = transform1
        self.transform_ = transform2
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['file'].values[idx]
        image = skimage.io.MultiImage(file_path)[0]
        image = cv2.resize(image,(SIZE,SIZE))
        label = self.df["label"].values[idx]
        origin_img = image
        if self.transform:
            image = self.transform(image=image)['image']
        if self.transform_:
            image = self.transform_(image=image)['image']

        
            
        label = torch.tensor(label).float()
        
        return image, label,origin_img
import torch.nn.functional as F
class SaveFeatures():
    """ Extract pretrained activations"""
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()


def getCAM_(feature_conv, weight_fc, class_idx):
    #Heatmap取得
    print("feature_conv",feature_conv)#None
    _, nc, h, w = feature_conv.shape
    cam = weight_fc.dot(feature_conv[0,:, :, ].reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img


def plotGradCAM(model, final_conv, fc_params, train_loader, 
                row=2, col=4, img_size=256, device='cuda', original=False):
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    # save activated_features from conv
    activated_features = SaveFeatures(final_conv)
    # save weight from fc
    weight = np.squeeze(fc_params[0].cpu().data.numpy())
    # original images
    if original:
        fig = plt.figure(figsize=(20, 15))
        for i, (img, target, org_img) in enumerate(train_loader):
            if i ==0:
                print(img.size())#bs,h,w,c
            img = img.permute(0, 3, 1, 2)
            if i ==0:
                print(img.size())#bs,c,h,w
            output = model(img.to(device))
            
            pred_idx = torch.sigmoid(output).to('cpu').numpy()
            if i ==0:
                print("元画像",org_img.size())#1, 512, 512, 3
            cur_images = org_img.numpy().transpose((0,1,2,3))
            if i ==0:
                print("cur_images",cur_images.shape)#1, 512, 3, 512
            ax = fig.add_subplot(row, col, i+1, xticks=[], yticks=[])
            plt.imshow(cv2.cvtColor(cur_images[0], cv2.COLOR_BGR2RGB))
            ax.set_title('Label:{0}, Predict:{1}'.format(target[0], pred_idx[0]), fontsize=14)
            if i == row*col-1:
                break
        plt.show()
    # heatmap images
    fig = plt.figure(figsize=(20, 15))
    for i, (img, target, _) in enumerate(train_loader):
        img = img.permute(0, 3, 1, 2)#bs,c,h,w
        if i ==0:
            print("val,img",img.size())#1, 3, 512, 512
            #print("check_label",target)
        output = model(img.to(device).float())
        pred_idx = torch.sigmoid(output).to('cpu').numpy()#0~1
        if i ==0:
            print("pred_idx",pred_idx)
        cur_images = img.cpu().numpy().transpose((0,2,3,1))
        if i ==0:
            print("val,cur_images",cur_images.shape)#1, 3, 512, 512
        #heatmap = getCAM(activated_features.features, weight, pred_idx)
        heatmap = getCAM(img, weight)
        ax = fig.add_subplot(row, col, i+1, xticks=[], yticks=[])
        plt.imshow(cv2.cvtColor(cur_images[0], cv2.COLOR_BGR2RGB))
        plt.imshow(cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_LINEAR), alpha=1, cmap='jet')
        ax.set_title('Label:{0}, Predict:{1}'.format(target[0], pred_idx[0]), fontsize=14)
        if i == row*col-1:
            break
    plt.show()

class_loaders = []
# we use fold=0 model for Grad-CAM
for fold in [0]:
    
    # idx
    val_idx = folds[folds['fold'] == fold].index # check by val data
    #val_idx = folds[folds['fold'] != fold].index # check by train data
    
    # prepare each label loader
    for i in range(2):
        valid_dataset = TrainDataset(folds.loc[val_idx][folds[CFG.target_col]==i].reset_index(drop=True),  
                                     transform1=get_transforms1(data='valid'))
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        class_loaders.append(valid_loader)
print(class_loaders[0])
"""
label = 1
weights_path = "/kaggle/working/fold{}_resnet18_baseline.pth".format(0)
model = baseline_model(weights_path)
plotGradCAM(model, final_conv, fc_params, class_loaders[label], img_size=SIZE, device=device, original=True)"""
"""
label = 0
weights_path = "/kaggle/working/fold{}_resnet18_baseline.pth".format(0)
model = baseline_model(weights_path)
plotGradCAM(model, final_conv, fc_params, class_loaders[label], img_size=SIZE, device=device, original=True)"""
