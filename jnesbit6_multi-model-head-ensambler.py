import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm,trange
from sklearn.model_selection import train_test_split
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
# exception
!pip install pytorchcv --quiet
from pytorchcv.model_provider import get_model
model_ex = get_model("xception", pretrained=True)
model_ex = nn.Sequential(*list(model_ex.children())[:-1]) # Remove original output layer
model_ex[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
model_ex.eval()
# resnext
import torch.nn as nn
import torchvision.models as models


class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3], 
                                        groups=32, 
                                        width_per_group=4)
        self.fc = nn.Linear(2048, 1)
checkpoint = torch.load("/kaggle/input/deepfakes-inference-demo/resnext.pth")

model_res = MyResNeXt().to('cuda')
model_res.load_state_dict(checkpoint)


del checkpoint

model_res = nn.Sequential(*list(model_res.children())[:-1]) # Remove original output layer
model_res.eval()
#model_res[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
# effiecientnet
import sys
package_path = '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master'
sys.path.append(package_path)

from efficientnet_pytorch import EfficientNet

class EfficientNet(EfficientNet):
    ...
    def extract_features_midconv(self, inputs):
        """ Returns output of the middle convolution layers """
        out_feats = []
        
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x_size = x.size()[-1]
        out_feats.append(x)
        
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if x.size()[-1] != prev_x_size:
                prev_x_size = x.size()[-1]
                out_feats.append(x)
            else:
                out_feats[-1] = x
        
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        if x.size()[-1] != prev_x_size:
            out_feats.append(x)
        else:
            out_feats[-1] = x
        
        return out_feats

weight_path = 'efficientnet_b0_epoch_15_loss_0.158.pth'
trained_weights_path = os.path.join('../input/deepfake-detection-model-weight', weight_path)

model_ef = EfficientNet.from_name('efficientnet-b0')
model_ef._fc = nn.Linear(in_features=model_ef._fc.in_features, out_features=1)
model_ef.load_state_dict(torch.load(trained_weights_path))
model_ef = model_ef.cuda()
model_ef = model_ef.extract_features
#model_ef = nn.Sequential(*list(model_ef.children())[:-1]) # Remove original output layer
#print(type(model_ef))
#model_ef.eval()


#model_ef[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))

class Head(torch.nn.Module):
  def __init__(self, in_f, out_f):
    super(Head, self).__init__()
    
    self.f = nn.Flatten()
    self.c = nn.Conv2d(1280, 2048, 1, [4, 4], 0)
    self.l = nn.Linear(in_f, 512)
    self.d = nn.Dropout(0.75)
    self.o = nn.Linear(512, out_f)
    self.b1 = nn.BatchNorm1d(in_f)
    self.b2 = nn.BatchNorm1d(512)
    self.r = nn.ReLU()

  def forward(self, x):
    
    x[0] = self.c(x[0])

    x = torch.cat((x[0], x[1], x[2]), 1)
    
    x = self.f(x)
    x = self.b1(x)
    x = self.d(x)

    x = self.l(x)
    x = self.r(x)
    x = self.b2(x)
    x = self.d(x)

    out = self.o(x)
    return out
class FCN(torch.nn.Module):
  def __init__(self, model_res, model_ef, model_ex, in_f):
    super(FCN, self).__init__()
    self.model_res = model_res
    self.model_ef = model_ef
    self.model_ex = model_ex
    self.h1 = Head(in_f, 1)
  
  def forward(self, x):
    x2 = self.model_res(x) # 1, 2048

    x3 = self.model_ex(x)

    x1 = self.model_ef(x)

    return self.h1([x1, x2, x3])

model = FCN(model_res, model_ef, model_ex, 6144)
model.eval()
model = model.cuda()
del model_res, model_ef, model_ex
import gc
model(torch.ones([1, 3, 150, 150]).cuda())
df_train0 = pd.read_json('../input/deepfake/metadata0.json')
df_train1 = pd.read_json('../input/deepfake/metadata1.json')
df_train2 = pd.read_json('../input/deepfake/metadata2.json')
df_train3 = pd.read_json('../input/deepfake/metadata3.json')
df_train4 = pd.read_json('../input/deepfake/metadata4.json')
df_train5 = pd.read_json('../input/deepfake/metadata5.json')
df_train6 = pd.read_json('../input/deepfake/metadata6.json')
df_train7 = pd.read_json('../input/deepfake/metadata7.json')
df_train8 = pd.read_json('../input/deepfake/metadata8.json')
df_train9 = pd.read_json('../input/deepfake/metadata9.json')
df_train10 = pd.read_json('../input/deepfake/metadata10.json')
df_train11 = pd.read_json('../input/deepfake/metadata11.json')
df_train12 = pd.read_json('../input/deepfake/metadata12.json')
df_train13 = pd.read_json('../input/deepfake/metadata13.json')
df_train14 = pd.read_json('../input/deepfake/metadata14.json')
df_train15 = pd.read_json('../input/deepfake/metadata15.json')
df_train16 = pd.read_json('../input/deepfake/metadata16.json')
df_train17 = pd.read_json('../input/deepfake/metadata17.json')
df_train18 = pd.read_json('../input/deepfake/metadata18.json')
df_train19 = pd.read_json('../input/deepfake/metadata19.json')
df_train20 = pd.read_json('../input/deepfake/metadata20.json')
df_train21 = pd.read_json('../input/deepfake/metadata21.json')
df_train22 = pd.read_json('../input/deepfake/metadata22.json')
df_train23 = pd.read_json('../input/deepfake/metadata23.json')
df_train24 = pd.read_json('../input/deepfake/metadata24.json')
df_train25 = pd.read_json('../input/deepfake/metadata25.json')
df_train26 = pd.read_json('../input/deepfake/metadata26.json')
df_train27 = pd.read_json('../input/deepfake/metadata27.json')
df_train28 = pd.read_json('../input/deepfake/metadata28.json')
df_train29 = pd.read_json('../input/deepfake/metadata29.json')
df_train30 = pd.read_json('../input/deepfake/metadata30.json')
df_train31 = pd.read_json('../input/deepfake/metadata31.json')
df_train32 = pd.read_json('../input/deepfake/metadata32.json')
df_train33 = pd.read_json('../input/deepfake/metadata33.json')
df_train34 = pd.read_json('../input/deepfake/metadata34.json')
df_train35 = pd.read_json('../input/deepfake/metadata35.json')
df_train36 = pd.read_json('../input/deepfake/metadata36.json')
df_train37 = pd.read_json('../input/deepfake/metadata37.json')
df_train38 = pd.read_json('../input/deepfake/metadata38.json')
df_train39 = pd.read_json('../input/deepfake/metadata39.json')
df_train40 = pd.read_json('../input/deepfake/metadata40.json')
df_train41 = pd.read_json('../input/deepfake/metadata41.json')
df_train42 = pd.read_json('../input/deepfake/metadata42.json')
df_train43 = pd.read_json('../input/deepfake/metadata43.json')
df_train44 = pd.read_json('../input/deepfake/metadata44.json')
df_train45 = pd.read_json('../input/deepfake/metadata45.json')
df_train46 = pd.read_json('../input/deepfake/metadata46.json')
df_val1 = pd.read_json('../input/deepfake/metadata47.json')
df_val2 = pd.read_json('../input/deepfake/metadata48.json')
df_val3 = pd.read_json('../input/deepfake/metadata49.json')
df_trains = [df_train0 ,df_train1, df_train2, df_train3, df_train4,
             df_train5, df_train6, df_train7, df_train8, df_train9,df_train10,
            df_train11, df_train12, df_train13, df_train14, df_train15,df_train16, 
            df_train17, df_train18, df_train19, df_train20, df_train21, df_train22, 
            df_train23, df_train24, df_train25, df_train26, df_train27, df_train28, 
            df_train29, df_train30, df_train31, df_train32, df_train33, df_train34,
            df_train34, df_train35, df_train36, df_train37, df_train38, df_train39,
            df_train40, df_train41, df_train42, df_train43, df_train44, df_train45,
            df_train46]
df_vals=[df_val1, df_val2, df_val3]
nums = list(range(len(df_trains)+1))
LABELS = ['REAL','FAKE']
val_nums=[47, 48, 49]
def get_path(num,x):
    num=str(num)
    if len(num)==2:
        path='../input/deepfake/DeepFake'+num+'/DeepFake'+num+'/' + x.replace('.mp4', '') + '.jpg'
    else:
        path='../input/deepfake/DeepFake0'+num+'/DeepFake0'+num+'/' + x.replace('.mp4', '') + '.jpg'
    if not os.path.exists(path):
       raise Exception
    return path
paths=[]
y=[]
for df_train,num in tqdm(zip(df_trains,nums),total=len(df_trains)):
    images = list(df_train.columns.values)
    for x in images:
        try:
            paths.append(get_path(num,x))
            y.append(LABELS.index(df_train[x]['label']))
        except Exception as err:
            #print(err)
            pass

val_paths=[]
val_y=[]
for df_val,num in tqdm(zip(df_vals,val_nums),total=len(df_vals)):
    images = list(df_val.columns.values)
    for x in images:
        try:
            val_paths.append(get_path(num,x))
            val_y.append(LABELS.index(df_val[x]['label']))
        except Exception as err:
            #print(err)
            pass
def read_img(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

def shuffle(X,y):
    new_train=[]
    for m,n in zip(X,y):
        new_train.append([m,n])
    random.shuffle(new_train)
    X,y=[],[]
    for x in new_train:
        X.append(x[0])
        y.append(x[1])
    return X,y

import random
def get_random_sampling(paths, y, val_paths, val_y):
  real=[]
  fake=[]
  for m,n in zip(paths,y):
      if n==0:
          real.append(m)
      else:
          fake.append(m)
  # fake=random.sample(fake,len(real))
  paths,y=[],[]
  for x in real:
      paths.append(x)
      y.append(0)
  for x in fake:
      paths.append(x)
      y.append(1)

  real=[]
  fake=[]
  for m,n in zip(val_paths,val_y):
      if n==0:
          real.append(m)
      else:
          fake.append(m)
  # fake=random.sample(fake,len(real))
  val_paths,val_y=[],[]
  for x in real:
      val_paths.append(x)
      val_y.append(0)
  for x in fake:
      val_paths.append(x)
      val_y.append(1)

  X=[]
  for img in tqdm(paths):
      X.append(read_img(img))
  val_X=[]
  for img in tqdm(val_paths):
      val_X.append(read_img(img))

  # Balance with ffhq dataset
  ffhq = os.listdir('../input/ffhq-face-data-set/thumbnails128x128')
  X_ = []
  for file in tqdm(ffhq):
    im = read_img(f'../input/ffhq-face-data-set/thumbnails128x128/{file}')
    im = cv2.resize(im, (150,150))
    X_.append(im)
  random.shuffle(X_)

  for i in range(64773 - 12130):
    X.append(X_[i])
    y.append(0)
  del X_[0:64773 - 12130]
  for i in range(6108 - 1258):
    val_X.append(X_[i])
    val_y.append(0)

  X, y = shuffle(X,y)
  val_X, val_y = shuffle(val_X,val_y)

  return X, val_X, y, val_y
from torch.utils.data import Dataset, DataLoader
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class ImageDataset(Dataset):
    def __init__(self, X, y, training=True, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.X[idx]

        if self.transform is not None:
          res = self.transform(image=img)
          img = res['image']
        
        img = np.rollaxis(img, 2, 0)
        # img = np.array(img).astype(np.float32) / 255.

        labels = self.y[idx]
        labels = np.array(labels).astype(np.float32)
        return [img, labels]
# model_res, model_ex, model_ef
# !pip install torchtoolbox --quiet
# from torchtoolbox.tools import summary

# model.cuda()
# summary(model, torch.rand((1, 3, 150, 150)).cuda())
def criterion1(pred1, targets):
  l1 = F.binary_cross_entropy(F.sigmoid(pred1), targets)
  return l1

def train_model(epoch, optimizer, scheduler=None, history=None):
    model.train()
    total_loss = 0
    
    t = tqdm(train_loader)
    for i, (img_batch, y_batch) in enumerate(t):
        
        img_batch = img_batch.cuda().float()
        y_batch = y_batch.cuda().float()

        optimizer.zero_grad()

        out = model(img_batch)
        loss = criterion1(out, y_batch)

        total_loss += loss
        t.set_description(f'Epoch {epoch+1}/{n_epochs}, LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(i+1)))

        if history is not None:
          history.loc[epoch + i / lenx, 'train_loss'] = loss.data.cpu().numpy()
          history.loc[epoch + i / lenx, 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']

        loss.backward()
        optimizer.step()
        if scheduler is not None:
          scheduler.step()

def evaluate_model(epoch, scheduler=None, history=None):
    model.eval()
    loss = 0
    pred = []
    real = []
    with torch.no_grad():
        for img_batch, y_batch in val_loader:
            img_batch = img_batch.cuda().float()
            y_batch = y_batch.cuda().float()

            o1 = model(img_batch)
            l1 = criterion1(o1, y_batch)
            loss += l1
            
            for j in o1:
              pred.append(F.sigmoid(j))
            for i in y_batch:
              real.append(i.data.cpu())
    
    pred = [p.data.cpu().numpy() for p in pred]
    pred2 = pred
    pred = [np.round(p) for p in pred]
    pred = np.array(pred)
    acc = sklearn.metrics.recall_score(real, pred, average='macro')

    real = [r.item() for r in real]
    pred2 = np.array(pred2).clip(0.1, 0.9)
    kaggle = sklearn.metrics.log_loss(real, pred2)

    loss /= len(val_loader)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
    
    if scheduler is not None:
      scheduler.step(loss)

    print(f'Dev loss: %.4f, Acc: %.6f, Kaggle: %.6f'%(loss,acc,kaggle))
    
    return loss
del df_trains
del df_vals
del df_train0 ,df_train1, df_train2, df_train3, df_train4,df_train5, df_train6, df_train7, df_train8, df_train9,df_train10,df_train11, df_train12, df_train13, df_train14, df_train15,df_train16, df_train17, df_train18, df_train19, df_train20, df_train21, df_train22, df_train23, df_train24, df_train25, df_train26, df_train27, df_train28, df_train29, df_train30, df_train31, df_train32, df_train33,df_train34, df_train35, df_train36, df_train37, df_train38, df_train39,df_train40, df_train41, df_train42, df_train43, df_train44, df_train45,df_train46
gc.collect()

X, val_X, y, val_y = get_random_sampling(paths, y, val_paths, val_y)

print('There are '+str(y.count(1))+' fake train samples')
print('There are '+str(y.count(0))+' real train samples')
print('There are '+str(val_y.count(1))+' fake val samples')
print('There are '+str(val_y.count(0))+' real val samples')
import albumentations
from albumentations.augmentations.transforms import ShiftScaleRotate, HorizontalFlip, Normalize, RandomBrightnessContrast, MotionBlur, Blur, GaussNoise, JpegCompression
train_transform = albumentations.Compose([
                                          ShiftScaleRotate(p=0.3, scale_limit=0.25, border_mode=1, rotate_limit=25),
                                          HorizontalFlip(p=0.2),
                                          RandomBrightnessContrast(p=0.3, brightness_limit=0.25, contrast_limit=0.5),
                                          MotionBlur(p=.2),
                                          GaussNoise(p=.2),
                                          JpegCompression(p=.2, quality_lower=50),
                                          Normalize()
])
val_transform = albumentations.Compose([
                                          Normalize()
])

train_dataset = ImageDataset(X, y, transform=train_transform)
val_dataset = ImageDataset(val_X, val_y, transform=val_transform)

lenx = len(X)
del X, y
gc.collect()
nrow, ncol = 5, 6
fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    image, label = train_dataset[i]
    image = np.rollaxis(image, 0, 3)
    image = image*std + mean
    image = np.clip(image, 0., 1.)
    ax.imshow(image)
    ax.set_title(f'label: {label}')
del axes


history = pd.DataFrame()
history2 = pd.DataFrame()

torch.cuda.empty_cache()
gc.collect()

best = 1e10
n_epochs = 2
batch_size = 80

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
del train_dataset
del val_dataset
gc.collect()

model = model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.7, verbose=True, min_lr=1e-5)

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    gc.collect()

    train_model(epoch, optimizer, scheduler=None, history=history)
    
    loss = evaluate_model(epoch, scheduler=scheduler, history=history2)
    
del train_loader
del val_loader
gc.collect() 

print(f'Saving best model...')
torch.save(model.h1.state_dict(), f'model.pth')
history2.plot()