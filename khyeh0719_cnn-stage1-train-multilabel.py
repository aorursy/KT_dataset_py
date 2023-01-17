package_path = '../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
import sys; sys.path.append(package_path)
!cp ../input/gdcm-conda-install/gdcm.tar .
!tar -xvzf gdcm.tar
!conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2
package_path = '../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
import sys; sys.path.append(package_path)

from glob import glob
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import cv2
import pydicom
from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

CFG = {
    'train': True,
    
    'train_img_path': '../input/rsna-str-pulmonary-embolism-detection/train',
    'test_img_path': '../input/rsna-str-pulmonary-embolism-detection/test',
    'cv_fold_path': '../input/stratified-validation-strategy/rsna_train_splits_fold_20.csv',
    'train_path': '../input/rsna-str-pulmonary-embolism-detection/train.csv',
    'test_path': '../input/rsna-str-pulmonary-embolism-detection/test.csv',
    
    'image_target_cols': [
        'pe_present_on_image', # only image level
        'rv_lv_ratio_gte_1', # exam level
        'rv_lv_ratio_lt_1', # exam level
        'leftsided_pe', # exam level
        'chronic_pe', # exam level
        'rightsided_pe', # exam level
        'acute_and_chronic_pe', # exam level
        'central_pe', # exam level
        'indeterminate' # exam level
    ],
    
    'img_size': 256,
    'lr': 0.0005,
    'epochs': 1,
    'device': 'cuda', # cuda, cpu
    'train_bs': 64,
    'valid_bs': 256,
    'accum_iter': 1,
    'verbose_step': 1,
    'num_workers': 4,
    'efbnet': 'efficientnet-b0',
    
    'train_folds': [np.arange(0,16),
                    #np.concatenate([np.arange(0,12), np.arange(16,20)]),
                    #np.concatenate([np.arange(0,8), np.arange(12,20)]),
                    #np.concatenate([np.arange(0,4), np.arange(8,20)]),
                    #np.arange(4,20),
                   ],#[np.arange(0,16)],
    
    'valid_folds': [np.arange(16,20),
                    #np.arange(12,16),
                    #np.arange(8,12),
                    #np.arange(4,8),
                    #np.arange(0,4)
                   ],#[np.arange(16,20)],
    
    'model_path': '../input/kh-rsna-model',
    'tag': 'efb0_stage1_multilabel',
}

SEED = 42321

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    #X = (X*255.0).astype('uint8')
    return X

def get_img(path):
    
    d = pydicom.read_file(path)
    '''
    res = cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (CFG['img_size'], CFG['img_size'])), d.ImagePositionPatient[2]
    '''
    
    '''
    RED channel / LUNG window / level=-600, width=1500
    GREEN channel / PE window / level=100, width=700
    BLUE channel / MEDIASTINAL window / level=40, width=400
    '''
    
    img = (d.pixel_array * d.RescaleSlope) + d.RescaleIntercept
    
    r = window(img, -600, 1500)
    g = window(img, 100, 700)
    b = window(img, 40, 400)
    
    res = np.concatenate([r[:, :, np.newaxis],
                          g[:, :, np.newaxis],
                          b[:, :, np.newaxis]], axis=-1)
    
    #res = (res*255.0).astype('uint8')
    res = zoom(res, [CFG['img_size']/res.shape[0], CFG['img_size']/res.shape[1], 1.], prefilter=False, order=1)
    #res = res.astype(np.float32)/255.
    
    return res

    '''
    
    img -= img.min()
    img /= img.max()
    return img[:, :, np.newaxis]
    '''

class RSNADataset(Dataset):
    def __init__(
        self, df, label_smoothing, data_root, 
        image_subsampling=True, transforms=None, output_label=True
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.label_smoothing = label_smoothing
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.df[CFG['image_target_cols']].iloc[index].values
            target[1:-1] = target[0]*target[1:-1] # if PE == 1, keep the original label; otherwise clean to 0 (except indeterminate)
            #print(target)
            
        path = "{}/{}/{}/{}.dcm".format(self.data_root, 
                                        self.df.iloc[index]['StudyInstanceUID'], 
                                        self.df.iloc[index]['SeriesInstanceUID'], 
                                        self.df.iloc[index]['SOPInstanceUID'])
        img  = get_img(path)
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        # do label smoothing
        if self.output_label == True:
            target = np.clip(target, self.label_smoothing, 1 - self.label_smoothing)
            
            return img, target
        else:
            return img

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, RandomResizedCrop
)

from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            #RandomRotate90(p=0.5),
            #HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            #RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=( -0.4, 0.4), p=0.5),
            #RandomResizedCrop(CFG['img_size'], CFG['img_size'], scale=(0.9, 1.0), ratio=(0.9, 1.1), p=1.0),
            #Cutout(num_holes=1, max_h_size=CFG['img_size']//2, max_w_size=CFG['img_size']//2, p=1.0),
            #Cutout(p=0.5),
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0),
        ], p=1.)
    '''
    return transforms.Compose([
            transforms.Lambda(lambda imgs: torch.stack([transforms.ToTensor()(img) for img in imgs])),
            transforms.Lambda(lambda imgs: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])(img) for img in imgs])),
           
        ])
    '''   
        
def get_valid_transforms():
    return Compose([
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0),
        ], p=1.)
    '''
    return transforms.Compose([
            transforms.Lambda(lambda imgs: torch.stack([transforms.ToTensor()(img) for img in imgs])),
            transforms.Lambda(lambda imgs: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])(img) for img in imgs])),
           
        ])
    '''  

class RNSAImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = EfficientNet.from_pretrained(CFG['efbnet'], in_channels=3)
        #print(self.cnn_model, CFG['efbnet'])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
    def get_dim(self):
        return self.cnn_model._fc.in_features
        
    def forward(self, x):
        feats = self.cnn_model.extract_features(x)
        return self.pooling(feats).view(x.shape[0], -1)                         

    
class RSNAImgClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = RNSAImageFeatureExtractor()
        self.image_predictors = nn.Linear(self.cnn_model.get_dim(), 9)
        
    def forward(self, imgs):
        #print(images.shape)
        imgs_embdes = self.cnn_model(imgs) # bs * efb_feat_size
        #print(imgs_embdes.shape)
        image_preds = self.image_predictors(imgs_embdes)
        
        return image_preds
    
#RSNAClassifier(64)
def rsna_wloss_inference(y_true_img, y_pred_img):
    bce_func = torch.nn.BCELoss(reduction='sum')
    image_loss = bce_func(y_pred_img, y_true_img)
    correct_count = ((y_pred_img>0) == y_true_img).sum()
    counts = y_pred_img.shape[0]
    return image_loss, correct_count, counts

def rsna_wloss_train(y_true_img, y_pred_img, device):
    bce_func = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device)
    y_pred_img = y_pred_img.view(*y_true_img.shape)
    image_loss = bce_func(y_pred_img, y_true_img)
    correct_count = ((y_pred_img>0) == y_true_img).sum(axis=0)
    counts = y_true_img.size()[0]
    
    return image_loss, correct_count, counts

def rsna_wloss_valid(y_true_img, y_pred_img, device):
    return rsna_wloss_train(y_true_img, y_pred_img, device)

def prepare_train_dataloader(train, cv_df, train_fold, valid_fold):
    from catalyst.data.sampler import BalanceClassSampler
    
    train_patients = cv_df.loc[cv_df.fold.isin(train_fold), 'StudyInstanceUID'].unique()
    valid_patients = cv_df.loc[cv_df.fold.isin(valid_fold), 'StudyInstanceUID'].unique()

    train_ = train.loc[train.StudyInstanceUID.isin(train_patients),:].reset_index(drop=True)
    valid_ = train.loc[train.StudyInstanceUID.isin(valid_patients),:].reset_index(drop=True)

    # train mode to do image-level subsampling
    train_ds = RSNADataset(train_, 0.0, CFG['train_img_path'],  image_subsampling=False, transforms=get_train_transforms(), output_label=True) 
    valid_ds = RSNADataset(valid_, 0.0, CFG['train_img_path'],  image_subsampling=False, transforms=get_valid_transforms(), output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=CFG['num_workers'],
        #sampler=BalanceClassSampler(labels=train_[CFG['image_target_cols'][0]].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    #print(len(train_loader), len(val_loader))

    return train_loader, val_loader

def train_one_epoch(epoch, model, device, scaler, optimizer, train_loader):
    model.train()

    t = time.time()
    loss_sum = 0
    acc_sum = None
    loss_w_sum = 0
    acc_record = []
    loss_record = []
    avg_cnt = 40
    
    for step, (imgs, image_labels) in enumerate(train_loader):
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).float()

        #print(image_labels.shape, exam_label.shape)
        with autocast():
            image_preds = model(imgs)   #output = model(input)
            #print(image_preds.shape, exam_pred.shape)

            image_loss, correct_count, counts = rsna_wloss_train(image_labels, image_preds, device)
            
            loss = image_loss/counts
            scaler.scale(loss).backward()

            loss_ = image_loss.detach().item()/counts
            acc_ = correct_count.detach().cpu().numpy()/counts
            
            loss_record += [loss_]
            acc_record += [acc_]
            loss_record = loss_record[-avg_cnt:]
            acc_record = acc_record[-avg_cnt:]
            loss_sum = np.vstack(loss_record).mean(axis=0)
            acc_sum = np.vstack(acc_record).mean(axis=0)
            
            #loss_w_sum += counts

            if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()                

            acc_details = ["{:.5}: {:.4f}".format(f, float(acc_sum[i])) for i, f in enumerate(CFG['image_target_cols'])]
            acc_details = ", ".join(acc_details)
            
            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                print(
                    f'epoch {epoch} train Step {step+1}/{len(train_loader)}, ' + \
                    f'loss: {loss_sum[0]:.3f}, ' + \
                    acc_details + ', ' + \
                    f'time: {(time.time() - t):.2f}', end='\r' if (step + 1) != len(train_loader) else '\n'
                )

def valid_one_epoch(epoch, model, device, scheduler, val_loader, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    acc_sum = None
    loss_w_sum = 0

    for step, (imgs, image_labels) in enumerate(val_loader):
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).float()
        
        image_preds = model(imgs)   #output = model(input)
        #print(image_preds.shape, exam_pred.shape)

        image_loss, correct_count, counts = rsna_wloss_valid(image_labels, image_preds, device)

        loss = image_loss/counts
        
        loss_sum += image_loss.detach().item()
        if acc_sum is None:
            acc_sum = correct_count.detach().cpu().numpy()
        else:
            acc_sum += correct_count.detach().cpu().numpy()
        loss_w_sum += counts     

        acc_details = ["{:.5}: {:.4f}".format(f, acc_sum[i]/loss_w_sum) for i, f in enumerate(CFG['image_target_cols'])]
        acc_details = ", ".join(acc_details)
            
        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            print(
                f'epoch {epoch} valid Step {step+1}/{len(val_loader)}, ' + \
                f'loss: {loss_sum/loss_w_sum:.3f}, ' + \
                acc_details + ', ' + \
                f'time: {(time.time() - t):.2f}', end='\r' if (step + 1) != len(val_loader) else '\n'
            )
    
    if schd_loss_update:
        scheduler.step(loss_sum/loss_w_sum)
    else:
        scheduler.step()
        
if __name__ == '__main__':
    if CFG['train']:
        from  torch.cuda.amp import autocast, GradScaler # for training only, need nightly build pytorch

    seed_everything(SEED)
    
    if CFG['train']:
        # read train file
        train_df = pd.read_csv(CFG['train_path'])

        # read cv file
        cv_df = pd.read_csv(CFG['cv_fold_path'])

        # img must be sorted before feeding into NN for correct orders
    else:
        #assert False, "This kernel is for training only!"
        # read test file
        test_df = pd.read_csv(CFG['test_path'])
    
    if CFG['train']:
        
        for fold, (train_fold, valid_fold) in enumerate(zip(CFG['train_folds'], CFG['valid_folds'])):
            if fold < 0:
                continue
            print(fold)   
            train_loader, val_loader = prepare_train_dataloader(train_df, cv_df, train_fold, valid_fold)

            device = torch.device(CFG['device'])
            model = RSNAImgClassifier().to(device)
            scaler = GradScaler()   
            optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1); schd_loss_update=True
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=1); schd_loss_update=False
            
            for epoch in range(CFG['epochs']):
                train_one_epoch(epoch, model, device, scaler, optimizer, train_loader)
                
                #model.load_state_dict(torch.load('{}/model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag'])))
                with torch.no_grad():
                    valid_one_epoch(epoch, model, device, scheduler, val_loader, schd_loss_update=schd_loss_update)
            
            #assert False
            
            torch.save(model.state_dict(),'{}/model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
            #torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
            del model, optimizer, train_loader, val_loader, scaler, scheduler
            torch.cuda.empty_cache()
            
        train_loader, val_loader = prepare_train_dataloader(train_df, cv_df, np.arange(0, 20), np.array([]))
        #print(len(train_loader), len(val_loader))
        device = torch.device(CFG['device'])
        model = RSNAImgClassifier().to(device)
        scaler = GradScaler()   
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1); schd_loss_update=True
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=1); schd_loss_update=False

        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, device, scaler, optimizer, train_loader)

        torch.save(model.state_dict(),'{}/model_{}'.format(CFG['model_path'], CFG['tag']))
        
    else:
        assert False