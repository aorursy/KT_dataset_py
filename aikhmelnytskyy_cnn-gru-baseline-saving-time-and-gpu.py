package_path = '../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'

import sys; sys.path.append(package_path)

!cp ../input/gdcm-conda-install/gdcm.tar .

!tar -xvzf gdcm.tar

!conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2
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

    'train': False,

    

    'train_img_path': '../input/rsna-str-pulmonary-embolism-detection/train',

    'test_img_path': '../input/rsna-str-pulmonary-embolism-detection/test',

    'cv_fold_path': '../input/stratified-validation-strategy/rsna_train_splits_fold_20.csv',

    'train_path': '../input/rsna-str-pulmonary-embolism-detection/train.csv',

    'test_path': '../input/rsna-str-pulmonary-embolism-detection/test.csv',

    

    'image_target_cols': [

        'pe_present_on_image', # only image level

    ],

    

    'exam_target_cols': [

        'negative_exam_for_pe', # exam level

        #'qa_motion',

        #'qa_contrast',

        #'flow_artifact',

        'rv_lv_ratio_gte_1', # exam level

        'rv_lv_ratio_lt_1', # exam level

        'leftsided_pe', # exam level

        'chronic_pe', # exam level

        #'true_filling_defect_not_pe',

        'rightsided_pe', # exam level

        'acute_and_chronic_pe', # exam level

        'central_pe', # exam level

        'indeterminate' # exam level

    ], 

    

    'img_num': 200,

    'img_size': 256,

    'lr': 0.0005,

    'epochs': 2,

    'device': 'cuda', # cuda, cpu

    'train_bs': 2,

    'accum_iter': 8,

    'verbose_step': 1,

    'num_workers': 4,

    'efbnet': 'efficientnet-b0',

    

    'train_folds': [np.arange(0,16),

                    np.concatenate([np.arange(0,12), np.arange(16,20)]),

                    np.concatenate([np.arange(0,8), np.arange(12,20)]),

                    np.concatenate([np.arange(0,4), np.arange(8,20)]),

                    np.arange(4,20),

                   ],#[np.arange(0,16)],

    

    'valid_folds': [np.arange(16,20),

                    np.arange(12,16),

                    np.arange(8,12),

                    np.arange(4,8),

                    np.arange(0,4)

                   ],#[np.arange(16,20)],

    

    'model_path': '../input/kh-rsna-model',

    'tag': 'efb0_stage2_multilabel'

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

    #res = cv2.resize(res, (CFG['img_size'], CFG['img_size']))

    #res = res.astype(np.float32)/255.

    

    return res



class RSNADatasetStage1(Dataset):

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

            target = self.df[CFG['image_target_cols']].values[index]

            

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

        

class RSNADataset(Dataset):

    def __init__(

        self, df, label_smoothing, data_root, 

        image_subsampling=True, transforms=None, output_label=True

    ):

        

        super().__init__()

        self.df = df

        self.patients = self.df['StudyInstanceUID'].unique()

        self.image_subsampling = image_subsampling

        self.label_smoothing = label_smoothing

        self.transforms = transforms

        self.data_root = data_root

        self.output_label = output_label

        

    def get_patients(self):

        return self.patients

        

    def __len__(self):

        return len(self.patients)

    

    def __getitem__(self, index: int):

        

        patient = self.patients[index]

        df_ = self.df.loc[self.df.StudyInstanceUID == patient]

        

        per_image_feats = get_stage1_columns()

        #print(per_image_feats)

        

        if self.image_subsampling:

            img_num = min(CFG['img_num'], df_.shape[0])

            

            # naive image subsampling

            img_ix = np.random.choice(np.arange(df_.shape[0]), replace=False, size=img_num)

            

            # get all images, then slice location and sort according to z values

            imgs = np.zeros((CFG['img_num'],), np.float32) #np.zeros((CFG['img_num'], CFG['img_size'], CFG['img_size'], 3), np.float32)

            per_image_preds = np.zeros((CFG['img_num'], len(per_image_feats)), np.float32)

            locs = np.zeros((CFG['img_num'],), np.float32)

            image_masks = np.zeros((CFG['img_num'],), np.float32)

            image_masks[:img_num] = 1.

            

            # get labels

            if self.output_label:

                exam_label = df_[CFG['exam_target_cols']].values[0]

                image_labels = np.zeros((CFG['img_num'], len(CFG['image_target_cols'])), np.float32)

            

        else:

            img_num = df_.shape[0]

            img_ix = np.arange(df_.shape[0])

            

            # get all images, then slice location and sort according to z values

            imgs = np.zeros((img_num, ), np.float32) #np.zeros((img_num, CFG['img_size'], CFG['img_size'], 3), np.float32)

            per_image_preds = np.zeros((img_num, len(per_image_feats)), np.float32)

            locs = np.zeros((img_num,), np.float32)

            image_masks = np.zeros((img_num,), np.float32)

            image_masks[:img_num] = 1.

            

            # get labels

            if self.output_label:

                exam_label = df_[CFG['exam_target_cols']].values[0]

                image_labels = np.zeros((img_num, len(CFG['image_target_cols'])), np.float32)

                

        for i, im_ix in enumerate(img_ix):

            path = "{}/{}/{}/{}.dcm".format(self.data_root, 

                                            df_['StudyInstanceUID'].values[im_ix], 

                                            df_['SeriesInstanceUID'].values[im_ix], 

                                            df_['SOPInstanceUID'].values[im_ix])

            

            d = pydicom.read_file(path)

            locs[i] = d.ImagePositionPatient[2]

            per_image_preds[i,:] = df_[per_image_feats].values[im_ix,:]

            

            if self.output_label == True:

                image_labels[i] = df_[CFG['image_target_cols']].values[im_ix]



        #print('get img done')

        

        seq_ix = np.argsort(locs)

        

        # image features: img_num * img_size * img_size * 1

        '''

        imgs = imgs[seq_ix]

        if self.transforms:

            imgs = [self.transforms(image=img)['image'] for img in imgs]

        imgs = torch.stack(imgs)

        '''

        

        # image level features: img_num

        #locs[:img_num] -= locs[:img_num].min()

        locs = locs[seq_ix]

        locs[1:img_num] = locs[1:img_num]-locs[0:img_num-1]

        locs[0] = 0

        

        per_image_preds = per_image_preds[seq_ix]

        

        # patient level features: 1

        

        # train, train-time valid, multiple patients: imgs, locs, image_labels, exam_label, img_num

        # whole valid-time valid, single patient: imgs, locs, image_labels, exam_label, img_num, sorted id

        # whole test-time test, single patient: imgs, locs, img_num, sorted_id

        

        # do label smoothing

        if self.output_label == True:

            image_labels = image_labels[seq_ix]

            image_labels = np.clip(image_labels, self.label_smoothing, 1 - self.label_smoothing)

            exam_label =  np.clip(exam_label, self.label_smoothing, 1 - self.label_smoothing)

            

            return imgs, per_image_preds, locs, image_labels, exam_label, image_masks

        else:

            return imgs, per_image_preds, locs, img_num, index, seq_ix



from albumentations import (

    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,

    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,

    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout

)



from albumentations.pytorch import ToTensorV2



def get_train_transforms():

    return Compose([

            #HorizontalFlip(p=0.5),

            #VerticalFlip(),

            #RandomRotate90(p=0.5),

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

        self.cnn_model = EfficientNet.from_name(CFG['efbnet'])

        #print(self.cnn_model, CFG['efbnet'])

        self.pooling = nn.AdaptiveAvgPool2d(1)

        

    def get_dim(self):

        return self.cnn_model._fc.in_features

        

    def forward(self, x):

        feats = self.cnn_model.extract_features(x)

        return self.pooling(feats).view(x.shape[0], -1)                         



class RSNAImgClassifierSingle(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn_model = RNSAImageFeatureExtractor()

        self.image_predictors = nn.Linear(self.cnn_model.get_dim(), 1)

        

    def forward(self, imgs):

        #print(images.shape)

        imgs_embdes = self.cnn_model(imgs) # bs * efb_feat_size

        #print(imgs_embdes.shape)

        image_preds = self.image_predictors(imgs_embdes)

        

        return image_preds

    

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



class TimeDistributed(nn.Module):



    def __init__(self, module, batch_first=True):

        super(TimeDistributed, self).__init__()

        self.module = module

        self.batch_first = batch_first



    def forward(self, x):

        ''' x size: (batch_size, time_steps, in_channels, height, width) '''

        x_size= x.size()

        c_in = x.contiguous().view(x_size[0] * x_size[1], *x_size[2:])

        

        c_out = self.module(c_in)

        r_in = c_out.view(x_size[0], x_size[1], -1)

        if self.batch_first is False:

            r_in = r_in.permute(1, 0, 2)

        return r_in 

    

class RSNAClassifier(nn.Module):

    def __init__(self, hidden_size=64):

        super().__init__()

        

        self.gru = nn.GRU(len(get_stage1_columns())+1, hidden_size, bidirectional=True, batch_first=True, num_layers=2)

        

        self.image_predictors = TimeDistributed(nn.Linear(hidden_size*2, 1))

        self.exam_predictor = nn.Linear(hidden_size*2*2, 9)

        

    def forward(self, img_preds, locs):

        

        embeds = torch.cat([img_preds, locs.view(locs.shape[0], locs.shape[1], 1)], dim=2) # bs * ts * fs

        

        embeds, _ = self.gru(embeds)

        image_preds = self.image_predictors(embeds)

        

        avg_pool = torch.mean(embeds, 1)

        max_pool, _ = torch.max(embeds, 1)

        conc = torch.cat([avg_pool, max_pool], 1)

        

        exam_pred = self.exam_predictor(conc)

        return image_preds, exam_pred

    

#RSNAClassifier(64)

def rsna_wloss_inference(y_true_img, y_true_exam, y_pred_img, y_pred_exam, chunk_sizes):

    # y_true_img, y_pred_img: (p1*in1 + p2*in2 + ,,,) 

    # y_true_exam, y_pred_exam: (p1*in1 + p2*in2 + ,,,) x 9

    # chunk_sizes: (patient_num)

    '''

    'negative_exam_for_pe', # exam level 0.0736196319

    'rv_lv_ratio_gte_1', # exam level 0.2346625767

    'rv_lv_ratio_lt_1', # exam level 0.0782208589

    'leftsided_pe', # exam level 0.06257668712

    'chronic_pe', # exam level 0.1042944785

    'rightsided_pe', # exam level 0.06257668712

    'acute_and_chronic_pe', # exam level 0.1042944785

    'central_pe', # exam level 0.1877300613

    'indeterminate' # exam level 0.09202453988

    '''

    

    # transform into torch tensors

    y_true_img, y_true_exam, y_pred_img, y_pred_exam = torch.tensor(y_true_img, dtype=torch.float32), torch.tensor(y_true_exam, dtype=torch.float32), torch.tensor(y_pred_img, dtype=torch.float32), torch.tensor(y_pred_exam, dtype=torch.float32)

    

    # split into chunks (each chunks is for a single exam)

    y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks = torch.split(y_true_img, chunk_sizes, dim=0), torch.split(y_true_exam, chunk_sizes, dim=0), torch.split(y_pred_img, chunk_sizes, dim=0), torch.split(y_pred_exam, chunk_sizes, dim=0)

    

    label_w = torch.tensor([0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988]).view(1, -1)

    img_w = 0.07361963

    bce_func = torch.nn.BCELoss(reduction='none')

    

    total_loss = torch.tensor(0, dtype=torch.float32)

    total_weights = torch.tensor(0, dtype=torch.float32)

    for i, (y_true_img_, y_true_exam_, y_pred_img_, y_pred_exam_) in enumerate(zip(y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks)):

        exam_loss = bce_func(y_pred_exam_[0, :], y_true_exam_[0, :])

        exam_loss = torch.sum(exam_loss*label_w, 1)[0] # Kaggle uses a binary log loss equation for each label and then takes the mean of the log loss over all labels.

        #print(exam_loss)



        image_loss = bce_func(y_pred_img_, y_true_img_)

        img_num = chunk_sizes[i]

        qi = torch.sum(y_true_img_)/img_num

        image_loss = torch.sum(img_w*qi*image_loss)

        #print(image_loss)

    

        total_loss += exam_loss+image_loss

        total_weights += label_w.sum() + img_w*qi*img_num

        #assert False

        

    final_loss = total_loss/total_weights

    return final_loss



def rsna_wloss_train(y_true_img, y_true_exam, y_pred_img, y_pred_exam, image_masks, device):

    # y_true_img, y_pred_img: patient_numximg_num

    # y_true_exam, y_pred_exam: patient_num x 9

    

    label_w = torch.tensor([0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988]).view(1, -1).to(device)

    img_w = 0.07361963

    bce_func = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

    

    total_loss = torch.tensor(0, dtype=torch.float32).to(device)

    total_weights = torch.tensor(0, dtype=torch.float32).to(device)

    for i in range(y_true_img.shape[0]):

        exam_loss = bce_func(y_pred_exam[i, :], y_true_exam[i, :])

        exam_loss = torch.sum(exam_loss*label_w, 1)[0] # Kaggle uses a binary log loss equation for each label and then takes the mean of the log loss over all labels.

        #print(exam_loss)



        img_mask = image_masks[i]

        #print(torch.sum(y_true_img[i,:]), torch.sum(img_mask))

        image_loss = bce_func(y_pred_img[i,:], y_true_img[i,:]).flatten()

        #print(image_loss.shape)

        #print(img_mask.shape)

        #print((image_loss*img_mask).shape)

        #assert False

        image_loss = image_loss*img_mask # mark 0 loss for padding images

        img_num = torch.sum(img_mask) #y_true_img.shape[1]

        qi = torch.sum(y_true_img[i,:])/img_num

        image_loss = torch.sum(img_w*qi*image_loss)

        #print(image_loss)

    

        total_loss += exam_loss+image_loss

        total_weights += label_w.sum() + img_w*qi*img_num

        #assert False

        

    final_loss = total_loss/total_weights

    return final_loss, total_loss, total_weights



def rsna_wloss_valid(y_true_img, y_true_exam, y_pred_img, y_pred_exam, image_masks, device):

    # y_true_img, y_pred_img: patient_numximg_num

    # y_true_exam, y_pred_exam: patient_num x 9

    

    label_w = torch.tensor([0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988]).view(1, -1).to(device)

    img_w = 0.07361963

    bce_func = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

    

    total_loss = torch.tensor(0, dtype=torch.float32).to(device)

    total_weights = torch.tensor(0, dtype=torch.float32).to(device)

    for i in range(y_true_img.shape[0]):

        exam_loss = bce_func(y_pred_exam[i, :], y_true_exam[i, :])

        exam_loss = torch.sum(exam_loss*label_w, 1)[0] # Kaggle uses a binary log loss equation for each label and then takes the mean of the log loss over all labels.

        #print(exam_loss)



        img_mask = image_masks[i]

        #print(torch.sum(y_true_img[i,:]), torch.sum(img_mask))

        image_loss = bce_func(y_pred_img[i,:], y_true_img[i,:]).flatten()

        #print(image_loss.shape)

        #print(img_mask.shape)

        #print((image_loss*img_mask).shape)

        #assert False

        image_loss = image_loss*img_mask # mark 0 loss for padding images

        img_num = torch.sum(img_mask) #y_true_img.shape[1]

        qi = torch.sum(y_true_img[i,:])/img_num

        image_loss = torch.sum(img_w*qi*image_loss)

        #print(image_loss)

    

        total_loss += exam_loss+image_loss

        total_weights += label_w.sum() + img_w*qi*img_num

        #assert False

        

    final_loss = total_loss/total_weights

    return final_loss, total_loss, total_weights



def prepare_train_dataloader(train, cv_df, train_fold, valid_fold):

    train_patients = cv_df.loc[cv_df.fold.isin(train_fold), 'StudyInstanceUID'].unique()

    valid_patients = cv_df.loc[cv_df.fold.isin(valid_fold), 'StudyInstanceUID'].unique()



    train_ = train.loc[train.StudyInstanceUID.isin(train_patients),:].reset_index(drop=True)

    valid_ = train.loc[train.StudyInstanceUID.isin(valid_patients),:].reset_index(drop=True)



    # train mode to do image-level subsampling

    train_ds = RSNADataset(train_, 0.0, CFG['train_img_path'],  image_subsampling=True, transforms=get_train_transforms(), output_label=True) 

    valid_ds = RSNADataset(valid_, 0.0, CFG['train_img_path'],  image_subsampling=False, transforms=get_valid_transforms(), output_label=True)



    train_loader = torch.utils.data.DataLoader(

        train_ds,

        batch_size=CFG['train_bs'],

        pin_memory=True,

        drop_last=False,

        shuffle=True,        

        num_workers=CFG['num_workers'],

    )

    val_loader = torch.utils.data.DataLoader(

        valid_ds, 

        batch_size=1,

        num_workers=CFG['num_workers'],

        shuffle=False,

        pin_memory=True,

    )

    #print(len(train_loader), len(val_loader))



    return train_loader, val_loader



def train_one_epoch(epoch, model, device, scaler, optimizer, train_loader):

    model.train()



    t = time.time()

    loss_sum = 0

    loss_w_sum = 0



    for step, (imgs, per_image_preds, locs, image_labels, exam_label, image_masks) in enumerate(train_loader):

        imgs = imgs.to(device).float()

        per_image_preds = per_image_preds.to(device).float()

        locs = locs.to(device).float()

        image_masks = image_masks.to(device).float()

        image_labels = image_labels.to(device).float()

        exam_label = exam_label.to(device).float()



        #print(image_labels.shape, exam_label.shape)

        with autocast():

            image_preds, exam_pred = model(per_image_preds, locs)   #output = model(input)

            #print(image_preds.shape, exam_pred.shape)



            loss, total_loss, total_weights = rsna_wloss_train(image_labels, exam_label, image_preds, exam_pred, image_masks, device)



            scaler.scale(loss).backward()



            loss_sum += total_loss.detach().item()

            loss_w_sum += total_weights.detach().item()



            if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):

                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)



                scaler.step(optimizer)

                scaler.update()

                optimizer.zero_grad()                



            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):

                print(

                    f'epoch {epoch} train step {step+1}/{len(train_loader)}, ' + \

                    f'loss: {loss_sum/loss_w_sum:.4f}, ' + \

                    f'time: {(time.time() - t):.4f}', end= '\r' if (step + 1) != len(train_loader) else '\n'

                )



def post_process(exam_pred, image_pred):

    

    rv_lv_ratio_lt_1_ix = CFG['exam_target_cols'].index('rv_lv_ratio_lt_1')

    rv_lv_ratio_gte_1_ix = CFG['exam_target_cols'].index('rv_lv_ratio_gte_1')

    central_pe_ix = CFG['exam_target_cols'].index('central_pe')

    rightsided_pe_ix = CFG['exam_target_cols'].index('rightsided_pe')

    leftsided_pe_ix = CFG['exam_target_cols'].index('leftsided_pe')

    acute_and_chronic_pe_ix = CFG['exam_target_cols'].index('acute_and_chronic_pe')

    chronic_pe_ix = CFG['exam_target_cols'].index('chronic_pe')

    negative_exam_for_pe_ix = CFG['exam_target_cols'].index('negative_exam_for_pe')

    indeterminate_ix = CFG['exam_target_cols'].index('indeterminate')

    

    # rule 1 or rule 2 judgement: if any pe image exist

    has_pe_image = torch.max(image_pred, 1)[0][0] > 0

    #print(has_pe_image)

    

    # rule 1-a: only one >= 0.5, the other < 0.5

    rv_lv_ratios = exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix]]

    rv_lv_ratios_1_a = nn.functional.softmax(rv_lv_ratios, dim=1) # to make one at least > 0.5

    rv_lv_ratios_1_a = torch.log(rv_lv_ratios_1_a/(1-rv_lv_ratios_1_a)) # turn back into logits

    exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix]] = torch.where(has_pe_image, rv_lv_ratios_1_a, rv_lv_ratios)

    

    # rule 1-b-1 or 1-b-2 judgement: at least one > 0.5

    crl_pe = exam_pred[:, [central_pe_ix, rightsided_pe_ix, leftsided_pe_ix]]

    has_no_pe = torch.max(crl_pe ,1)[0] <= 0 # all <= 0.5

    #print(has_no_pe)

    #assert False

        

    # rule 1-b

    max_val = torch.max(crl_pe, 1)[0]

    crl_pe_1_b = torch.where(crl_pe==max_val, 0.0001-crl_pe+crl_pe, crl_pe)

    exam_pred[:, [central_pe_ix, rightsided_pe_ix, leftsided_pe_ix]] = torch.where(has_pe_image*has_no_pe, crl_pe_1_b, crl_pe)

    

    # rule 1-c-1 or 1-c-2 judgement: at most one > 0.5

    ac_pe = exam_pred[:, [acute_and_chronic_pe_ix, chronic_pe_ix]]

    both_ac_ch = torch.min(ac_pe ,1)[0] > 0 # all > 0.5

    

    # rule 1-c

    ac_pe_1_c = nn.functional.softmax(ac_pe, dim=1) # to make only one > 0.5

    ac_pe_1_c = torch.log(ac_pe_1_c/(1-ac_pe_1_c)) # turn back into logits

    exam_pred[:, [acute_and_chronic_pe_ix, chronic_pe_ix]] = torch.where(has_pe_image*both_ac_ch, ac_pe_1_c, ac_pe)

    

    # rule 1-d

    neg_ind = exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]]

    neg_ind_1d = torch.clamp(neg_ind, max=0)

    exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]] = torch.where(has_pe_image, neg_ind_1d, neg_ind)

    

    # rule 2-a

    ne_inde = exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]]

    ne_inde_2_a = nn.functional.softmax(ne_inde, dim=1) # to make one at least > 0.5

    ne_inde_2_a = torch.log(ne_inde_2_a/(1-ne_inde_2_a)) # turn back into logits

    exam_pred[:, [negative_exam_for_pe_ix, indeterminate_ix]] = torch.where(~has_pe_image, ne_inde_2_a, ne_inde)

    

    # rule 2-b

    all_other_exam_labels = exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix,

                                          central_pe_ix, rightsided_pe_ix, leftsided_pe_ix,

                                          acute_and_chronic_pe_ix, chronic_pe_ix]]

    all_other_exam_labels_2_b = torch.clamp(all_other_exam_labels, max=0)

    exam_pred[:, [rv_lv_ratio_lt_1_ix, rv_lv_ratio_gte_1_ix,

                  central_pe_ix, rightsided_pe_ix, leftsided_pe_ix,

                  acute_and_chronic_pe_ix, chronic_pe_ix]] = torch.where(~has_pe_image, all_other_exam_labels_2_b, all_other_exam_labels)

    

    return exam_pred, image_pred

    

def valid_one_epoch(epoch, model, device, scheduler, val_loader, schd_loss_update=False):

    model.eval()



    t = time.time()

    loss_sum = 0

    loss_w_sum = 0



    for step, (imgs, per_image_preds, locs, image_labels, exam_label, image_masks) in enumerate(val_loader):

        imgs = imgs.to(device).float()

        per_image_preds = per_image_preds.to(device).float()

        locs = locs.to(device).float()

        image_masks = image_masks.to(device).float()

        image_labels = image_labels.to(device).float()

        exam_label = exam_label.to(device).float()



        #print(image_labels.shape, exam_label.shape)

        #with autocast():

        image_preds, exam_pred = model(per_image_preds, locs)   #output = model(input)

        #print(image_preds.shape, exam_pred.shape)

        exam_pred, image_preds= post_process(exam_pred, image_preds)



        loss, total_loss, total_weights = rsna_wloss_valid(image_labels, exam_label, image_preds, exam_pred, image_masks, device)



        loss_sum += total_loss.detach().item()

        loss_w_sum += total_weights.detach().item()          



        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):

            print(

                f'epoch {epoch} valid Step {step+1}/{len(val_loader)}, ' + \

                f'loss: {loss_sum/loss_w_sum:.4f}, ' + \

                f'time: {(time.time() - t):.4f}', end='\r' if (step + 1) != len(val_loader) else '\n'

            )

    

    if schd_loss_update:

        scheduler.step(loss_sum/loss_w_sum)

    else:

        scheduler.step()



def check_label_consistency(checking_df):

    # CHECKING CONSISTENCY OF POSITIVE EXAM LABELS

    df = checking_df.copy()

    print(df.shape)

    df['positive_images_in_exam'] = df['StudyInstanceUID'].map(df.groupby(['StudyInstanceUID']).pe_present_on_image.max())



    df_pos = df.loc[df.positive_images_in_exam >  0.5]

    df_neg = df.loc[df.positive_images_in_exam <= 0.5]



    rule1a = df_pos.loc[((df_pos.rv_lv_ratio_lt_1  >  0.5)  & 

                         (df_pos.rv_lv_ratio_gte_1 >  0.5)) | 

                        ((df_pos.rv_lv_ratio_lt_1  <= 0.5)  & 

                         (df_pos.rv_lv_ratio_gte_1 <= 0.5))].reset_index(drop = True)

    rule1a['broken_rule'] = '1a'



    rule1b = df_pos.loc[(df_pos.central_pe    <= 0.5) & 

                        (df_pos.rightsided_pe <= 0.5) & 

                        (df_pos.leftsided_pe  <= 0.5)].reset_index(drop = True)

    rule1b['broken_rule'] = '1b'



    rule1c = df_pos.loc[(df_pos.acute_and_chronic_pe > 0.5) & 

                        (df_pos.chronic_pe           > 0.5)].reset_index(drop = True)

    rule1c['broken_rule'] = '1c'

    # CHECKING CONSISTENCY OF NEGATIVE EXAM LABELS



    rule1d = df_pos.loc[(df_pos.indeterminate        > 0.5) | 

                        (df_pos.negative_exam_for_pe > 0.5)].reset_index(drop = True)

    rule1d['broken_rule'] = '1d'



    rule2a = df_neg.loc[((df_neg.indeterminate        >  0.5)  & 

                         (df_neg.negative_exam_for_pe >  0.5)) | 

                        ((df_neg.indeterminate        <= 0.5)  & 

                         (df_neg.negative_exam_for_pe <= 0.5))].reset_index(drop = True)

    rule2a['broken_rule'] = '2a'



    rule2b = df_neg.loc[(df_neg.rv_lv_ratio_lt_1     > 0.5) | 

                        (df_neg.rv_lv_ratio_gte_1    > 0.5) |

                        (df_neg.central_pe           > 0.5) | 

                        (df_neg.rightsided_pe        > 0.5) | 

                        (df_neg.leftsided_pe         > 0.5) |

                        (df_neg.acute_and_chronic_pe > 0.5) | 

                        (df_neg.chronic_pe           > 0.5)].reset_index(drop = True)

    rule2b['broken_rule'] = '2b'

    # MERGING INCONSISTENT PREDICTIONS

    errors = pd.concat([rule1a, rule1b, rule1c, rule1d, rule2a, rule2b], axis = 0)

    

    print('label in-consistency counts:', errors.shape)

        

    if errors.shape[0] > 0:

        print(errors.broken_rule.value_counts())

        print(errors)

        assert False

        

def inference(model, device, df, root_path):

    model.eval()



    t = time.time()



    ds = RSNADataset(df, 0.0, root_path,  image_subsampling=False, transforms=get_valid_transforms(), output_label=False)

    

    dataloader = torch.utils.data.DataLoader(

        ds, 

        batch_size=1,

        num_workers=CFG['num_workers'],

        shuffle=False,

        pin_memory=True,

    )

    

    patients = ds.get_patients()

    

    res_dfs = []

    

    for step, (imgs, per_image_preds, locs, img_num, index, seq_ix) in enumerate(dataloader):

        imgs = imgs.to(device).float()

        per_image_preds = per_image_preds.to(device).float()

        locs = locs.to(device).float()

        

        index = index.detach().numpy()[0]

        seq_ix = seq_ix.detach().numpy()[0,:]

        

        patient_filt = (df.StudyInstanceUID == patients[index])

        

        patient_df = pd.DataFrame()

        patient_df['SOPInstanceUID'] = df.loc[patient_filt, 'SOPInstanceUID'].values[seq_ix]

        patient_df['SeriesInstanceUID'] = df.loc[patient_filt, 'SeriesInstanceUID'].values # no need to sort

        patient_df['StudyInstanceUID'] = patients[index] # single value

        

        for c in CFG['image_target_cols']+CFG['exam_target_cols']:

            patient_df[c] = 0.0



        #with autocast():

        image_preds, exam_pred = model(per_image_preds, locs)   #output = model(input)

        #print(image_preds.shape, exam_pred.shape)

        

        exam_pred, image_preds = post_process(exam_pred, image_preds)

        

        exam_pred = torch.sigmoid(exam_pred).cpu().detach().numpy()

        image_preds = torch.sigmoid(image_preds).cpu().detach().numpy()



        patient_df[CFG['exam_target_cols']] = exam_pred[0]

        patient_df[CFG['image_target_cols']] = image_preds[0,:]

        res_dfs += [patient_df]



        '''

        res_df = res_df.merge(patient_df, on=['SOPInstanceUID', 'StudyInstanceUID'], how='left')

        '''

        # naive slow version

        '''

        res_df.loc[patient_filt, CFG['exam_target_cols']] = exam_pred[0]

        for si, sop_id in enumerate(sop_ids):

            sop_filt = (patient_filt) & (res_df.SOPInstanceUID == sop_id)

            res_df.loc[sop_filt, CFG['image_target_cols']] = image_preds[0, si]

        '''

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(dataloader)):

            print(

                f'Inference Step {step+1}/{len(dataloader)}, ' + \

                f'time: {(time.time() - t):.4f}', end='\r' if (step + 1) != len(dataloader) else '\n'

            )

                

    res_dfs = pd.concat(res_dfs, axis=0).reset_index(drop=True)

    res_dfs = df[['SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID']].merge(res_dfs, on=['SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID'], how='left')

    print(res_dfs[CFG['image_target_cols']+CFG['exam_target_cols']].head(5))

    print(res_dfs[CFG['image_target_cols']+CFG['exam_target_cols']].tail(5))

    assert res_dfs.shape[0] == df.shape[0]

    check_label_consistency(res_dfs)

    

    return res_dfs

  

STAGE1_CFGS = [

    {

        'tag': 'efb0_stage1',

        'model_constructor': RSNAImgClassifierSingle,

        'dataset_constructor': RSNADatasetStage1,

        'output_len': 1

    },

    {

        'tag': 'efb0_stage1_multilabel',

        'model_constructor': RSNAImgClassifier,

        'dataset_constructor': RSNADatasetStage1,

        'output_len': 9

    },

]

STAGE1_CFGS_TAG = 'efb0-stage1-single-multi-label'





def get_stage1_columns():

    

    new_feats = []

    for cfg in STAGE1_CFGS:

        for i in range(cfg['output_len']):

            f = cfg['tag']+'_'+str(i)

            new_feats += [f]

        

    return new_feats



def update_stage1_oof_preds(df, cv_df):

    

    res_file_name = STAGE1_CFGS_TAG+"-train.csv"    

    

    new_feats = get_stage1_columns()

    for f in new_feats:

        df[f] = 0

    

    if os.path.isfile(res_file_name):

        df = pd.read_csv(res_file_name)

        print('img acc:', ((df[new_feats[0]]>0)==df[CFG['image_target_cols'][0]]).mean())

        return df

    

    

    for fold, (train_fold, valid_fold) in enumerate(zip(CFG['train_folds'], CFG['valid_folds'])):

        if fold < 0:

            continue

            

        valid_patients = cv_df.loc[cv_df.fold.isin(valid_fold), 'StudyInstanceUID'].unique()

        filt = df.StudyInstanceUID.isin(valid_patients)

        valid_ = df.loc[filt,:].reset_index(drop=True)



        image_preds_all_list = []

        for cfg in STAGE1_CFGS:

            valid_ds = cfg['dataset_constructor'](valid_, 0.0, CFG['train_img_path'],  image_subsampling=False, transforms=get_valid_transforms(), output_label=True)



            val_loader = torch.utils.data.DataLoader(

                valid_ds, 

                batch_size=256,

                num_workers=CFG['num_workers'],

                shuffle=False,

                pin_memory=False,

                sampler=SequentialSampler(valid_ds)

            )



            device = torch.device(CFG['device'])

            model = cfg['model_constructor']().to(device)

            model.load_state_dict(torch.load('{}/model_fold_{}_{}'.format(CFG['model_path'], fold, cfg['tag'])))

            model.eval()



            image_preds_all = []

            correct_count = 0

            count = 0

            for step, (imgs, target) in enumerate(val_loader):

                imgs = imgs.to(device).float()

                target = target.to(device).float()



                image_preds = model(imgs)   #output = model(input)

                #print(image_preds[:,0], image_preds[:,0].shape)

                #print(target, target.shape)

                

                if len(image_preds.shape) == 1:

                    image_preds = image_preds.view(-1, 1)

                

                correct_count += ((image_preds[:,0]>0) == target[:,0]).sum().detach().item()

                count += imgs.shape[0]

                image_preds_all += [image_preds.cpu().detach().numpy()]

                print('acc: {:.4f}, {}, {}, {}/{}'.format(correct_count/count, correct_count, count, step+1, len(val_loader)), end='\r')

            print()

            

            image_preds_all = np.concatenate(image_preds_all, axis=0)

            image_preds_all_list += [image_preds_all]

        

            del model, val_loader

            torch.cuda.empty_cache()

        

        image_preds_all_list = np.concatenate(image_preds_all_list, axis=1)

        df.loc[filt, new_feats] = image_preds_all_list

        

    df.to_csv(res_file_name, index=False)

    return df

            

def update_stage1_test_preds(df):

    

    new_feats = get_stage1_columns()

    for f in new_feats:

        df[f] = 0

    

    image_preds_all_list = []

    for cfg in STAGE1_CFGS:

        test_ds = cfg['dataset_constructor'](df, 0.0, CFG['test_img_path'],  image_subsampling=False, transforms=get_valid_transforms(), output_label=False)



        test_loader = torch.utils.data.DataLoader(

            test_ds, 

            batch_size=256,

            num_workers=CFG['num_workers'],

            shuffle=False,

            pin_memory=False,

            sampler=SequentialSampler(test_ds)

        )



        device = torch.device(CFG['device'])

        model = cfg['model_constructor']().to(device)

        model.load_state_dict(torch.load('{}/model_{}'.format(CFG['model_path'], cfg['tag'])))

        model.eval()



        image_preds_all = []

        for step, imgs in enumerate(tqdm(test_loader)):

            imgs = imgs.to(device).float()



            image_preds = model(imgs)   #output = model(input)

            image_preds_all += [image_preds.cpu().detach().numpy()]

            #print(imgs[0], image_preds[0,:]); break

        

        #continue

        image_preds_all = np.concatenate(image_preds_all, axis=0)

        image_preds_all_list += [image_preds_all]

        

        del model, test_loader

        torch.cuda.empty_cache()

        

    image_preds_all_list = np.concatenate(image_preds_all_list, axis=1)

    df.loc[:,new_feats] = image_preds_all_list

    

    return df



if __name__ == '__main__':

    if CFG['train']:

        from  torch.cuda.amp import autocast, GradScaler # for training only, need nightly build pytorch



    seed_everything(SEED)

    

    if CFG['train']:

        # read train file

        train_df = pd.read_csv(CFG['train_path'])



        # read cv file

        cv_df = pd.read_csv(CFG['cv_fold_path'])



        with torch.no_grad():

            train_df = update_stage1_oof_preds(train_df,cv_df)

        # img must be sorted before feeding into NN for correct orders

    else:

        #assert False, "This kernel is for training only!"

        # read test file

        from os import path

        do_full=False

        if path.exists('../input/rsna-str-pulmonary-embolism-detection/train') and not do_full:

            test_df=pd.read_csv(CFG['test_path']).head(50)

        else:

            test_df=pd.read_csv(CFG['test_path'])

        #test_df = pd.read_csv(CFG['test_path'])

        

        with torch.no_grad():

            test_df = update_stage1_test_preds(test_df)

    

    if CFG['train']:

        for fold, (train_fold, valid_fold) in enumerate(zip(CFG['train_folds'], CFG['valid_folds'])):



            train_loader, val_loader = prepare_train_dataloader(train_df, cv_df, train_fold, valid_fold)



            device = torch.device(CFG['device'])

            model = RSNAClassifier().to(device)

            

            scaler = GradScaler()   

            optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])

            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1); schd_loss_update=True

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=1); schd_loss_update=False



            for epoch in range(CFG['epochs']):

                train_one_epoch(epoch, model, device, scaler, optimizer, train_loader)



                with torch.no_grad():

                    valid_one_epoch(epoch, model, device, scheduler, val_loader, schd_loss_update=schd_loss_update)



            torch.save(model.state_dict(),'{}/model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))

            

            model.load_state_dict(torch.load('{}/model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag'])))

            

            # debug

            #valid_one_epoch(1, model, device, scheduler, val_loader, schd_loss_update=schd_loss_update)

            

            # prediction for oof

            valid_patients = cv_df.loc[cv_df.fold.isin(valid_fold), 'StudyInstanceUID'].unique()

            valid_ = train_df.loc[train_df.StudyInstanceUID.isin(valid_patients),:].reset_index(drop=True)



            with torch.no_grad():

                val_pred_df = inference(model, device, valid_, CFG['train_img_path'])

            

            target = valid_[CFG['image_target_cols']].values

            pred = (val_pred_df[CFG['image_target_cols']].values > 0.5).astype(int)

            print('Image PE Accuracy: {:.3f}'.format((target==pred).mean()*100))

            

            loss = rsna_wloss_inference(valid_[CFG['image_target_cols']].values, valid_[CFG['exam_target_cols']].values, 

                                        val_pred_df[CFG['image_target_cols']].values, val_pred_df[CFG['exam_target_cols']].values, 

                                        list(valid_.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count()))



            print('Validation loss = {:.4f}'.format(loss.detach().item()))

            

            del model, optimizer, train_loader, val_loader, scaler, scheduler

            torch.cuda.empty_cache()

            

        train_loader, val_loader = prepare_train_dataloader(train_df, cv_df, np.arange(0, 20), np.array([]))

        #print(len(train_loader), len(val_loader))

        device = torch.device(CFG['device'])

        model = RSNAClassifier().to(device)

        scaler = GradScaler()   

        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])

        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1); schd_loss_update=True

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=1); schd_loss_update=False



        for epoch in range(CFG['epochs']):

            train_one_epoch(epoch, model, device, scaler, optimizer, train_loader)



        torch.save(model.state_dict(),'{}/model_{}'.format(CFG['model_path'], CFG['tag']))

        

    else:

        device = torch.device(CFG['device'])

        model = RSNAClassifier().to(device)

        model.load_state_dict(torch.load('{}/model_{}'.format(CFG['model_path'], CFG['tag'])))

        test_pred_df = inference(model, device, test_df, CFG['test_img_path'])       

        test_pred_df.to_csv('kh_submission_raw.csv')

        

        # transform into submission format

        ids = []

        labels = []

        

        gp_mean = test_pred_df.loc[:, ['StudyInstanceUID']+CFG['exam_target_cols']].groupby('StudyInstanceUID', sort=False).mean()

        for col in CFG['exam_target_cols']:

            ids += [[patient+'_'+col for patient in gp_mean.index]]

            labels += [gp_mean[col].values]

            

        ids += [test_pred_df.SOPInstanceUID.values]

        labels += [test_pred_df[CFG['image_target_cols']].values[:,0]]

        ids = np.concatenate(ids)

        labels = np.concatenate(labels)

        

        assert len(ids) == len(labels)

        

        submission = pd.DataFrame()

        submission['id'] = ids

        submission['label'] = labels

        print(submission.head(3))

        print(submission.tail(3))

        print(submission.shape)

        submission.to_csv('submission.csv', index=False)