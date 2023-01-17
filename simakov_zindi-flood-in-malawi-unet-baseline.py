import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random 

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


from scipy.interpolate import griddata
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable

import os
import copy
import gc
import cv2
from datetime import datetime

from albumentations.pytorch import ToTensor
from albumentations import (OneOf, PadIfNeeded, OpticalDistortion,
                            GridDistortion, ElasticTransform, GaussianBlur,
                            MedianBlur, MotionBlur, Compose,
                            ShiftScaleRotate, VerticalFlip, HorizontalFlip)

import warnings
warnings.filterwarnings("ignore")
# metaparemeters
n_folds = 5
seed = 0
data_path = '../input/zindi-flood-prediction/'
path_models = './models/'
if not os.path.exists(path_models):
    os.makedirs(path_models)
train = pd.read_csv(os.path.join(data_path, 'Train.csv'))
ss = pd.read_csv(os.path.join(data_path, 'SampleSubmission.csv'))
# get all unique x, y coordinates
_x = train['X'].round(2).unique()
_y = train['Y'].round(2).unique()
print(len(_x), len(_y))

#and create their all possible combinations
_xn = np.meshgrid(_x, _y)[0]
_yn = np.meshgrid(_x, _y)[1]

all_xy = np.dstack([_xn, _yn]).reshape(-1, 2)


#with this combinations create an empty df and merge it with the original one by unique (x, y) tuples
train_full = pd.DataFrame()
train_full.loc[:, 'X'] = all_xy[:, 0]
train_full.loc[:, 'Y'] = all_xy[:, 1]
train_full.loc[:, 'id'] = train_full['X'].astype(str) + '_' + train_full['Y'].astype(str)
id2ind = dict(zip(train_full.loc[:, 'id'].values, train_full.loc[:, 'id'].factorize()[0]))
train_full.loc[:, 'id'] = train_full.loc[:, 'id'].map(id2ind)

train.loc[:, 'id'] = train['X'].astype(str) + '_' + train['Y'].astype(str)
train.loc[:, 'id'] = train.loc[:, 'id'].map(id2ind)
del train['X'], train['Y']

train_full = train_full.merge(train, on=['id'], how='left').sort_values(['Y', 'X'], ascending=[False, True]).reset_index(drop=True)
del train_full['id']
#sanity check that we can switch from IMG to DF and vice versa
def df2pix(ind):
    assert  ind < 161 * 144
    h = np.floor(ind / 161)
    w = ind - h * 161
    return int(h), int(w)

def pix2df(h, w):
    assert h < 144
    assert w < 161
    ind = h * 161 + w
    return int(ind)

img = train_full['elevation'].values.reshape(144,161)
print(f'img: 50:55, {img.flatten()[50:55]}')
print(f'df: 50:55, {train_full["elevation"].values[50:55]}')
print(f'df2img: loc 3000, {train_full["elevation"].loc[3000]} -> {img[df2pix(3000)]}')
print(f'img2df: (34, 46), {img[34, 46]} -> {train_full["elevation"].loc[pix2df(34, 46)]}')
#functions for filling NaN by max falue in each channel and min-max normalization
def _fill_na(img):
    _img = img.copy()
    if len(img.shape) == 3:
        for i in range(_img.shape[2]):
            _img[np.isnan(_img[:, :, i]), i] = np.max(_img[~np.isnan(_img[:, :, i]), i])
    else:
        _img[np.isnan(_img)] = np.max(_img[~np.isnan(_img)])
    return _img

def _norm(img):
    _img = img.copy()
    if len(img.shape) == 3:
        for i in range(_img.shape[2]):
            _img[:, :, i] = (_img[:, :, i] - _img[:, :, i].min()) / (_img[:, :, i].max() - _img[:, :, i].min())  
    else:
        _img = (_img - _img.min()) / (_img.max() - _img.min())
    return _img
    
#create simple feature

rain2019 = train_full[['precip 2019-01-20 - 2019-01-27', 'precip 2019-01-27 - 2019-02-03',
       'precip 2019-02-03 - 2019-02-10', 'precip 2019-02-10 - 2019-02-17',
       'precip 2019-02-17 - 2019-02-24', 'precip 2019-02-24 - 2019-03-03',
       'precip 2019-03-03 - 2019-03-10', 'precip 2019-03-10 - 2019-03-17',
       'precip 2019-03-17 - 2019-03-24', 'precip 2019-03-24 - 2019-03-31',
       'precip 2019-03-31 - 2019-04-07', 'precip 2019-04-07 - 2019-04-14',
       'precip 2019-04-14 - 2019-04-21', 'precip 2019-04-21 - 2019-04-28',
       'precip 2019-04-28 - 2019-05-05', 'precip 2019-05-05 - 2019-05-12',
       'precip 2019-05-12 - 2019-05-19']].sum(axis=1)

rain2015 = train_full[['precip 2014-12-28 - 2015-01-04',
       'precip 2015-01-04 - 2015-01-11', 'precip 2015-01-11 - 2015-01-18',
       'precip 2015-01-18 - 2015-01-25', 'precip 2015-01-25 - 2015-02-01',
       'precip 2015-02-01 - 2015-02-08', 'precip 2015-02-08 - 2015-02-15',
       'precip 2015-02-15 - 2015-02-22', 'precip 2015-02-22 - 2015-03-01',
       'precip 2015-03-01 - 2015-03-08', 'precip 2015-03-08 - 2015-03-15',]].sum(axis=1)


img1 = train_full['elevation'].values.reshape(144,161)
img2 = train_full['LC_Type1_mode'].values.reshape(144,161)
img3_train = np.log1p(rain2015).values.reshape(144,161)
img3_test = np.log1p(rain2019).values.reshape(144,161)

img_target = train_full['target_2015'].values.reshape(144,161)
img_train = np.dstack([img1, img2, img3_train])
img_test = np.dstack([img1, img2, img3_test])

mask = ~np.isnan(img_target)
plt.figure(figsize=(10, 6))

plt.subplot(2, 3, 1)
plt.imshow(img_target)
plt.title('2015 target')

plt.subplot(2, 3, 2)
plt.imshow(img_train[:, :, 0])
plt.title('elevation')

plt.subplot(2, 3, 3)
plt.imshow(img_train[:, :, 1])
plt.title('LC_Type1_mode')

plt.subplot(2, 3, 4)
plt.imshow(img_train[:, :, 2])
plt.title('2015 rains')

plt.subplot(2, 3, 5)
plt.imshow(img_test[:, :, 2])
plt.title('2019 rains')

plt.subplot(2, 3, 6)
plt.imshow(mask)
plt.title('mask')


plt.show()
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + 2)
    random.seed(seed + 4)
       
class SnapshotEns(object):
    def __init__(self, k):
        self.best_loss = np.array([np.inf]*k)
        self.k = k
        self.models = [0]*k

    def train(self, model, loss):
        if np.any(self.best_loss > loss):
            presorted_loss = self.best_loss
            self.best_loss = np.sort(self.best_loss)
            
            self.models = [self.models[z] for z in np.argsort(presorted_loss)]
        
            pos = np.where(self.best_loss > loss)[0][-1]

            self.best_loss[pos] = loss            
            self.models[pos] = copy.deepcopy(model.eval())

    def get(self):
        return (self.best_loss, self.models)
    
    def predict(self, data1):
        preds = 0
        for model in self.models:
            model.eval()
            preds += model.forward(data1)
        preds /= self.k
        return preds

def rmse(y, pred, mask):
    loss = np.sqrt(np.sum((_fill_na(y)*mask - pred*mask)**2) / np.sum(mask))
    return loss

class Trainer():
    def __init__(self, net, net_params, opt, opt_params, criterion, n_epochs, device,
                 dataloader=None, is_snap=False, snap_k=3,
                 sch=None, scheduler_params=None, verbose=1):
        self.net = net
        self.net_params = net_params
        self.opt = opt
        self.opt_params = opt_params
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.device = device
        self.is_snap = is_snap
        self.snap_k = snap_k
        self.dataloader = dataloader
        self.sch = sch
        self.scheduler_params = scheduler_params
        self.verbose = verbose
        
    def fit(self, dataloader=None):
        if dataloader is not None:
            self.dataloader = dataloader
        if self.dataloader is None and dataloader is None:
            raise ValueError('At least one dataloader should be not None')
            
        
        self.model = self.net(**self.net_params).to(self.device)
        self.optimizer = self.opt(self.model.parameters(), **self.opt_params)   
        if self.sch is not None:
            self.scheduler = self.sch(self.optimizer, 'min', **self.scheduler_params)
            
        self.se = SnapshotEns(k=self.snap_k)
        train_log = []
        for epoch in range(self.n_epochs):
            # train
            train_loss = self.train()
            train_log.extend(train_loss)   
            # test
            val_loss, val_data = self.test()
            if self.is_snap: 
                self.se.train(self.model, np.mean(val_loss))
            if (self.verbose is not None) and ((epoch + 1) % self.verbose == 0):
                print('Epoch: {e}, train loss: {tl}, val loss: {vl}, val metric: {rmse}'.format(rmse=rmse(*val_data),
                                                                                                e=epoch,
                                                                                                tl=np.mean(train_loss),
                                                                                                vl=np.mean(val_loss)))
            if self.sch is not None:
                self.scheduler.step(np.mean(val_loss))
                
        if self.is_snap:
            val_loss, val_data = self.test(snap=True)
            print('Result, val loss: {vl}, val metric: {rmse}'.format(rmse=rmse(*val_data),
                                                                      vl=np.mean(val_loss)))
            
    def train(self):        
        loss_log =  []
        self.model.train()
        running_loss = 0
        for n, sample in enumerate(self.dataloader['train']):
            
            image = Variable(sample['image']).to(self.device)
            mask = Variable((sample['image1'].mean(dim=1) > 0)).to(self.device).float()
            y_train = Variable((sample['image2'].mean(dim=1))).to(self.device).float()
                    
            output = self.model(image).squeeze(1)
            loss = self.criterion(output * mask, y_train * mask) / mask.sum()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            
            loss = loss.data.cpu().numpy()
            loss_log.append(loss)
            running_loss += loss
            
        return loss_log
    
    def test(self, snap=False):        
        loss_log =  []
        self.model.eval()
        with torch.no_grad():
            for n, sample in enumerate(self.dataloader['val']):
                image = Variable(sample['image']).to(self.device)
                mask = Variable((sample['image1'].mean(dim=1) > 0)).to(self.device).float()
                y_val = Variable((sample['image2'].mean(dim=1))).to(self.device).float()

                if snap:
                    output = self.se.predict(image).squeeze(1)
                else:
                    output = self.model(image).squeeze(1)
                loss = self.criterion(output * mask, y_val * mask) / mask.sum()
                loss = loss.data.cpu().numpy()
                loss_log.append(loss)
            
        return loss_log, (y_val[0].data.cpu().numpy(), output[0].data.cpu().numpy(), mask[0].data.cpu().numpy())
    
    def predict(self, X):
        X_test = _norm(_fill_na(X.copy()).astype(np.float32))
        self.model.eval()
        with torch.no_grad():
            sample = self.dataloader['test'].dataset.transforms['test'](**{'image': X_test})
            image = Variable(sample['image']).unsqueeze(0).to(self.device)
            if self.is_snap:
                output = self.se.predict(image).squeeze(1)
            else:
                output = self.model(image).squeeze(1)
            
        return output[0].data.cpu().numpy()
    

# pytorch dataset
class OneShotSegDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, mask, transforms, stage, steps):
        #transform target and mask to rgb for some augmentations
        self.X = _norm(_fill_na(X.copy()).astype(np.float32))
        self.y = cv2.cvtColor(_fill_na(y.copy()).astype(np.float32), cv2.COLOR_GRAY2RGB).copy()
        self.mask = cv2.cvtColor((mask.copy()).astype(np.float32), cv2.COLOR_GRAY2RGB).copy()
        
        self.transforms = transforms
        self.stage = stage
        self.steps = steps
        
    def __len__(self):
        return steps 

    def __getitem__(self, index):       
        sample = self.transforms[self.stage](**{'image': self.X, 'image1': self.mask, 'image2': self.y})                  
        return sample
    
def is_shuffle(stage):
    is_sh = {'train': True, 'val': False, 'test': False}    
    return is_sh[stage]
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                  nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):

    def __init__(self, in_channels, n_class):
        super(UNet, self).__init__()
        self.dconv_down1 = DoubleConv(in_channels, 6)
        self.dconv_down2 = DoubleConv(6, 9)      

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = DoubleConv(9, 12)
        self.dconv_up2 = DoubleConv(9 + 12, 9)
        self.dconv_up1 = DoubleConv(9 + 6, 6)
        
        self.conv_last = nn.Conv2d(6, n_class, 1)
            
       
    def forward(self, x):
        inp = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
                
        x = self.dconv_up3(x)
        x = self._upsample_cat(x, conv2)       

        x = self.dconv_up2(x)
        x = self._upsample_cat(x, conv1)  
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return self._check_shape(out, inp)
    
    def _check_shape(self, x, y):

        _, _, H, W = y.size()
        if y.shape != x.shape:
            return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        else:
            return x
    
    def _upsample_add(self, x, y):
        return self._check_shape(x, y) + y

        
    def _upsample_cat(self, x, y):
        return torch.cat([self._check_shape(x, y), y], dim=1)
#create folds
train_full.loc[:, 'folds'] = 0
not_na_index = train_full.index[~train_full.target_2015.isna()]
folds = list(KFold(n_splits=n_folds, random_state=seed, shuffle=False).split(not_na_index))
for n, (tr, vl) in enumerate(folds):
    train_full.loc[not_na_index[vl], 'folds'] = n + 1 
val_folds = train_full['folds'].values.reshape(144, 161) 
plt.figure(figsize=(10, 5))
for i in range(n_folds):
    plt.subplot(2, n_folds, i+1)
    plt.imshow(mask*(1-(val_folds==(i+1))))
    plt.title(f'Training fold {i}')
    
    plt.subplot(2, n_folds, n_folds+i+1)
    plt.imshow(mask*(val_folds==(i+1)))
    plt.title(f'Validation fold {i}')

plt.tight_layout()
plt.show()  
#set augmentations
augs = {'train' : Compose([VerticalFlip(p=0.5), HorizontalFlip(p=0.5),
                           MotionBlur(blur_limit=5, p=0.05),
                           OneOf([ElasticTransform(alpha=1, sigma=10, alpha_affine=20,
                                                   interpolation=1, border_mode=2, p=1),
                                  GridDistortion(num_steps=3, distort_limit=0.3,
                                                 interpolation=1, border_mode=4, p=1),
                                  OpticalDistortion(distort_limit=0.2, shift_limit=0.2,
                                                    interpolation=1, border_mode=4, p=1)],
                                 p=0.1),
                           ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=45,
                                            interpolation=1, border_mode=4, p=0.5),
                           ToTensor()],
                          additional_targets={'image1': 'image', 'image2': 'image'}),

        'val' : Compose([ToTensor()],
                        additional_targets={'image1': 'image', 'image2': 'image'}),

        'test': Compose([ToTensor()],
                        additional_targets={'image1': 'image', 'image2': 'image'})}
#check, that image, target and mask transformed in the same way
set_seed(1)
plt.figure(figsize=(20, 15))
for i in range(3):
    sample = augs['train'](**{'image':  _norm(_fill_na(img_train)),
                              'image1':  cv2.cvtColor(_fill_na(mask.copy()).astype(np.float32),
                                                      cv2.COLOR_GRAY2RGB),
                              'image2':  cv2.cvtColor(_fill_na(img_target.copy()).astype(np.float32),
                                                      cv2.COLOR_GRAY2RGB)})
    plt.subplot(3, 3, i + 1)
    plt.imshow(sample['image'].numpy()[0])
    plt.title(f"Training image {i}. First channel only")
    
    plt.subplot(3, 3, 3 + i + 1)
    plt.imshow(sample['image1'].numpy()[0])
    plt.title(f"Mask image {i}")
    
    plt.subplot(3, 3, 6 + i + 1)
    plt.imshow(sample['image2'].numpy()[0])
    plt.title(f"Target image {i}")
    
plt.tight_layout()
plt.show()    
#more aug examples
set_seed(42)
plt.figure(figsize=(20, 15))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(augs['train'](image = _norm(_fill_na(img_train)))['image'].numpy()[0])
    plt.title(f"Training image {i}. First channel only")
    
plt.tight_layout()
plt.show()    
%%time
set_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#init model and training params
bs = 1 #batch size
num_workers = 4
steps = 50 #images per epoch
n_epochs = 50
snap_k = 5 # number of snapshots per fold
n_starts = 2

nn_model = Trainer(net=UNet,
                   net_params={'in_channels': 3, 'n_class': 1},
                   opt=torch.optim.Adam,
                   opt_params={'lr':1e-3},
                   criterion=nn.MSELoss(reduction='sum'),
                   n_epochs=n_epochs,
                   device=torch.device("cuda:0"),
                   is_snap=True,
                   snap_k=snap_k,
                   sch=lr_scheduler.ReduceLROnPlateau,
                   scheduler_params={'patience':15, 'factor':0.5, 'verbose':False},
                   verbose=None)



#Start training
oof = np.zeros((144, 161))

for j in range(n_starts):
    print('=====================')
    print(f'Iteration: {j}')
    for i in range(n_folds):
        print(f'Fold: {i}')
        val_folds = train_full['folds'].values.reshape(144, 161)

        masks = {'train': mask*(1-(val_folds==(i+1))), 'val': mask*(val_folds==(i+1)), 'test': mask}
        
        image_datasets = {x: OneShotSegDataset(X={'train': img_train, 'val': img_train, 'test': img_test}[x],
                                               y=img_target, mask=masks[x], transforms=augs, stage=x,
                                               steps=steps) for x in ['train', 'val', 'test']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size={'train': bs, 'val': 1, 'test': 1}[x],
                                                      shuffle=is_shuffle(x),
                                                      num_workers=num_workers,
                                                      pin_memory=True) for x in ['train', 'val', 'test']}
        
        nn_model.fit(dataloaders)

        pred = nn_model.predict(img_train)        
        oof += mask * (val_folds==(i+1)) * pred
        print('')
        
        torch.save(nn_model, os.path.join(path_models, f'model_iter_{j}_fold_{i}.pickle'))

    print(f'OOF rmse: {rmse(img_target, oof / (j+1), mask)}')
    print('')

oof /= (j+1)
oof = np.zeros((144, 161))
pred_train = np.zeros(144 * 161)
pred_test = np.zeros(144 * 161)


for j in range(n_starts):
    for i in range(n_folds):
        nn_model = torch.load(os.path.join(path_models, f'model_iter_{j}_fold_{i}.pickle'))
        
        pred = nn_model.predict(img_train)
        pred_train += pred.flatten() / n_folds / n_starts
        pred_test += nn_model.predict(img_test).flatten() / n_folds / n_starts        
        oof += mask * (val_folds==(i+1)) * pred

oof /= (j+1)
print(f'OOF rmse: {rmse(img_target, oof, mask)}')
print('')        
ss_full = train_full[['Square_ID']].copy()
ss_full.loc[:, 'target_2019'] = pred_train
ss_full = ss_full[~ss_full.Square_ID.isna()].reset_index(drop=True)
ss_full.to_csv('sub_train.csv', index=None) # ~0.11 public lb +- 0.01
ss_full.head()
ss_full = train_full[['Square_ID']].copy()
ss_full.loc[:, 'target_2019'] = oof.flatten()
ss_full = ss_full[~ss_full.Square_ID.isna()].reset_index(drop=True)
ss_full.to_csv('sub_train_oof.csv', index=None)  # ~0.15 public lb +- 0.01
ss_full.head()
ss_full = train_full[['Square_ID']].copy()
ss_full.loc[:, 'target_2019'] = pred_test
ss_full = ss_full[~ss_full.Square_ID.isna()].reset_index(drop=True)
ss_full.to_csv('sub_test.csv', index=None)  # ~0.11 public lb +- 0.01
ss_full.head()
plt.figure(figsize=(15, 3))

plt.subplot(1, 4, 1)
plt.imshow(img_target)
plt.title('2015 target')

plt.subplot(1, 4, 2)
plt.imshow(np.ma.masked_where(1-mask, pred_train.reshape(144, 161)))
plt.title('Train prediction')

plt.subplot(1, 4, 3)
plt.imshow(np.ma.masked_where(1-mask, pred_test.reshape(144, 161)))
plt.title('Test prediction')

plt.subplot(1, 4, 4)
plt.imshow(np.ma.masked_where(1-mask, oof))
plt.title('Train OOF prediction')

plt.tight_layout()
plt.show()
def get_data(train, color):
    x, y, z, c = train['X'], train['Y'], train['elevation'], color
    xi=np.linspace(min(x), max(x),200)
    yi=np.linspace(min(y),max(y),200)
    X,Y= np.meshgrid(xi,yi)
    Z = (griddata((x,y), z, (X, Y), method='nearest'))
    C = (griddata((x,y), c, (X, Y), method='nearest'))
    return X, Y, Z, C
    
def plot3D(X, Y, Z, C):
    data = [go.Surface(x=X,y=Y, z=Z, surfacecolor=C ,colorscale='Viridis')]
    layout = go.Layout(
        width=800,
        height=900,
        autosize=False,
        margin=dict(t=0, b=0, l=0, r=0),
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230, 230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectratio = dict(x=1, y=1, z=0.7),
            aspectmode = 'manual'
        )
    )

    updatemenus=list([
        dict(
            buttons=list([
                dict(
                    args=['type', 'surface'],
                    label='3D Surface',
                    method='restyle'
                ),
                
            ]),
            direction = 'left',
            pad = {'r': 10, 't': 10},
            showactive = True,
            type = 'buttons',
            x = 0.1,
            xanchor = 'left',
            y = 1.1,
            yanchor = 'top'
        ),
    ])

    annotations = list([
        dict(text='Trace type:', x=0, y=1.085, yref='paper', align='left', showarrow=False)
    ])
    layout['updatemenus'] = updatemenus
    layout['annotations'] = annotations

    fig = dict(data=data, layout=layout)
    iplot(fig, filename='cmocean-picker-one-button')
    
plot3D(*get_data(train_full, train_full['target_2015']))
plot3D(*get_data(train_full, oof.flatten()))