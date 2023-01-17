!pip -q install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

!pip -q install geffnet
import os

import sys

import time



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



import cv2

import PIL.Image



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from tqdm.notebook import tqdm



from sklearn.metrics import roc_auc_score, roc_curve, auc



# PyTorch elements

import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset # Using to construct Customed Datasets

import torch.nn as nn

import torch.nn.functional as F



from torch.utils.data import random_split



import torchvision

from torchvision import models

import torchvision.transforms as transforms

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler



# Scheduler

from torch.optim import lr_scheduler

from torch.optim.lr_scheduler import CosineAnnealingLR

from warmup_scheduler import GradualWarmupScheduler  # https://github.com/ildoonet/pytorch-gradual-warmup-lr



# Augmentation

import albumentations as A

import geffnet



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
DEBUG         = True

# Config

kernel_type   = 'efficientNet_Lite_256_meta_ext_9classes_5epochs'



org_data_dir  = '../input/jpeg-melanoma-256x256'

ext_data_dir  = '../input/jpeg-isic2019-256x256'



image_size    = 256

num_workers   = 4  # Using while loading data



batch_size    = 64

out_dim       = 9  # Output dims of CNN models

n_epochs      = 2

learning_rate = 3e-5



n_TTA         = 5
df_train = pd.read_csv(os.path.join(org_data_dir, 'train.csv'))

df_train.sample(7)
df_train.diagnosis.value_counts()
# Duplicates are rows with 'tfrecord' == -1

df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)



# 'jpg_path'

df_train['jpg_path'] = df_train['image_name'].apply(lambda image_name: os.path.join(org_data_dir, 'train', f'{image_name}.jpg'))



# 'diagnosis': matching 2020 (original) diagnosis to 2019(external) diagnosis

df_train['diagnosis'] = df_train['diagnosis'].apply(lambda string : string.replace('seborrheic keratosis', 'BKL'))

df_train['diagnosis'] = df_train['diagnosis'].apply(lambda string : string.replace('lichenoid keratosis', 'BKL'))

df_train['diagnosis'] = df_train['diagnosis'].apply(lambda string : string.replace('solar lentigo', 'BKL'))

df_train['diagnosis'] = df_train['diagnosis'].apply(lambda string : string.replace('lentigo NOS', 'BKL'))

df_train['diagnosis'] = df_train['diagnosis'].apply(lambda string : string.replace('cafe-au-lait macule', 'unknown'))

df_train['diagnosis'] = df_train['diagnosis'].apply(lambda string : string.replace('atypical melanocytic proliferation', 'unknown'))

df_train['diagnosis'].value_counts()
df_train_ext = pd.read_csv(os.path.join(ext_data_dir, 'train.csv'))

df_train_ext
# Duplicates are rows with 'tfrecord' == -1

df_train_ext = df_train_ext[df_train_ext['tfrecord'] != -1].reset_index(drop=True)



# 'jpg_path'

df_train_ext['jpg_path'] = df_train_ext['image_name'].apply(lambda image_name: os.path.join(ext_data_dir, 'train', f'{image_name}.jpg'))



# 'diagnosis': matching 2020 (original) diagnosis to 2019(external) diagnosis

df_train_ext['diagnosis'] = df_train_ext['diagnosis'].apply(lambda string : string.replace('NV', 'nevus'))

df_train_ext['diagnosis'] = df_train_ext['diagnosis'].apply(lambda string: string.replace('MEL', 'melanoma'))



df_train = pd.concat([df_train, df_train_ext]).reset_index(drop=True)
# Label encoding MANUALLY

# to assure what 'melanoma' index is

diagnosis_to_idx = {diag : idx for idx, diag in enumerate(np.sort(df_train.diagnosis.unique()))}

df_train['target'] = df_train['diagnosis'].map(diagnosis_to_idx)



melanoma_idx = diagnosis_to_idx['melanoma'] # We would use mel_idx latter



print("'diagnosis' encoding: \n",diagnosis_to_idx)

print('\n melanoma_idx: ', melanoma_idx)
df_train['target'].value_counts().plot(kind ='bar', 

                                       title='Counts: Target by Diagnosis')
# 'anatom_site_general_challenge': One-hot encode

tmp_dummies = pd.get_dummies(df_train['anatom_site_general_challenge'], prefix='site_', dummy_na=True)

df_train = pd.concat([df_train, tmp_dummies], axis=1)



# 'sex': male = 1, female = 0

df_train['sex'] = df_train['sex'].map({'male':   0,

                                       'female': 1})

df_train['sex'].fillna(-1, inplace=True)



# 'age_approx': max normalize

max_age = np.max(df_train['age_approx'])

df_train['age_approx'] /= max_age

df_train['age_approx'].fillna(0, inplace=True)



df_train['patient_id'].fillna(0, inplace=True)



# 'n_img' per user

map_PatientID_to_N_image_name = df_train.groupby(['patient_id']).image_name.count()

df_train['n_images'] = df_train['patient_id'].map(map_PatientID_to_N_image_name)

df_train.loc[df_train['patient_id']==-1, 'n_images'] = 1

df_train['n_images'] = np.log1p(df_train.n_images.values)



# 'image_size'

train_images = df_train['jpg_path'].values

tmp_train_sizes = np.zeros(train_images.shape[0])

for i, img_path in enumerate(tqdm(train_images)):

    tmp_train_sizes[i] = os.path.getsize(img_path)



df_train['image_size'] = np.log(tmp_train_sizes) # Logarit normalize



# Sum up

meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]

n_meta_features = len(meta_features)
df_train.sample(7)
n_meta_features
df_debug = df_train.sample(batch_size * 3)
class img_meta_Dataset(torch.utils.data.Dataset):

    def __init__(self, df_data=df_train, transformer=None):

        self.df_data = df_data

        self.transformer = transformer

        

    def __len__(self):

        return self.df_data.shape[0]

    

    def __getitem__(self, idx):

        row = self.df_data.iloc[idx]

        img = cv2.imread(row['jpg_path'])

        meta = self.df_data.iloc[idx][meta_features]

        # Augmentation

        if self.transformer is not None:

            img = self.transformer(image=img)  ## Transformer returns: {'images': np.array[list of Images]}

            img = img['image'].astype(np.float32)

        else:

            img = img.astype(np.float32)

        # Covert to 3 Channels * Hight * Weight format

        img = img.transpose(2, 0, 1)

        

        ## To torch.tensor.float32

        tensor_img = torch.tensor(img).float()

        tensor_metadata = torch.tensor(meta).float()

        x = (tensor_img, tensor_metadata)

        target = torch.tensor(row['target']).long()

        

        return (x, target)
# img_meta_Dataset works?

'''

df_debug = df_train.sample(5)

df_debug = img_meta_Dataset(df_debug, transformer=None)

dataloader = DataLoader(df_debug, batch_size=1)



for x, target in dataloader:

    x_img, x_meta = x

    print('x_img.shape: ',x_img.shape)

    print('x_meta.shape: ',x_meta.shape)



    print('target: {} \n'.format(target))

'''
augmentation = A.Compose([

    A.Transpose(p=0.5),

    A.HorizontalFlip(p=05.),

    A.VerticalFlip(p=0.5),

    A.RandomBrightness(p=0.75, limit=0.2),

    A.RandomContrast(p=0.75, limit=0.2),   

    A.OneOf([

        A.MotionBlur(blur_limit=5),

        A.MedianBlur(blur_limit=5),

        A.GaussianBlur(blur_limit=5),

        A.GaussNoise(var_limit=(0.5, 30))],

        p=0.7),

    A.OneOf([

        A.OpticalDistortion(distort_limit=1.0),

        A.GridDistortion(num_steps=5, distort_limit=1),

        A.ElasticTransform(alpha=3)],

        p=0.7),

    

    A.CLAHE(clip_limit=4.0, p=0.7),

    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),

    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),

    A.Resize(image_size, image_size),

    A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),    

    A.Normalize()

])
aug_resize_norm = A.Compose([

    A.Resize(image_size, image_size),

    A.Normalize(),

])
# Augmentation works?

df_debug = df_train.sample(batch_size * 3)

df_debug = img_meta_Dataset(df_debug, transformer=augmentation)



for plots in range(2):

    f, axis = plt.subplots(1, 5, figsize=(20, 20))

    for subplot in range(5):

        idx = np.random.randint(0, len(df_debug))

        x, target = df_debug[idx]

        x_img, _ = x

        axis[subplot].imshow(x_img.transpose(0, 1).transpose(1, 2).squeeze())

        axis[subplot].set_title(str(target))
criterion = nn.CrossEntropyLoss()
# Predicting with BOTH images and metadata

class blood_sweat_tears(nn.Module):

    def __init__(self, out_dim=out_dim, n_meta_features=n_meta_features):

        super().__init__()

        self.n_meta_features = n_meta_features

        

        self.effnet = geffnet.create_model('efficientnet_lite0', pretrained=True)

        

        in_ch = self.effnet.classifier.in_features + 128 # 128 is below meta_data_model's out_dim 

        self.final_fc = nn.Linear(in_ch, out_dim)

        

        self.effnet.classifier = nn.Identity()

        self.dropout = nn.Dropout(0.5) # Using while concatenating predictions, dropout = 0.5

    

    def img_model(self, x_img):       # To predict basedon IMAGES only

        x_img = self.effnet(x_img)

        return x_img

        

    def meta_data_model(self, x_meta): # To predict basedon META only

        self.meta_model = nn.Sequential(

                    nn.Linear(self.n_meta_features, 512), # 74 x 512

                    nn.BatchNorm1d(512),

                    nn.ReLU(),

                    nn.Dropout(p=0.3),

                    nn.Linear(512, 128),

                    nn.BatchNorm1d(128),

                    nn.ReLU(),)

        self.meta_model = self.meta_model.to(device)

        x_meta = self.meta_model(x_meta)

        return x_meta

    

    def forward(self, x_img, x_meta):  # Torch's mandatory

        x_img  = self.img_model(x_img).squeeze(-1).squeeze(-1) ####

        x_meta = self.meta_data_model(x_meta)

        # Concatenate BOTH predictions from images & meta

        x      = torch.cat((x_img, x_meta), dim=1)

        x      = self.dropout(x)

        x      = self.final_fc(x)

        # x      = x.softmax(1)  

        # NOT an activation exists here, as CrossEntropyLoss may prefer a logit input.

        return x
# Check: if MODELS works?

'''

df_debug = df_train.sample(batch_size * 3)

df_debug = img_meta_Dataset(df_debug, transformer=augmentation)

dataloader = DataLoader(df_debug, batch_size=3)



model = blood_sweat_tears(out_dim=out_dim)

model = model.to(device)



loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss = []



model.train()



bar = tqdm(dataloader)

for (x, target) in bar:

    optimizer.zero_grad()

    # print(f'target {target}')

    x_img, x_meta = x

    x_img, x_meta, target = x_img.to(device), x_meta.to(device), target.to(device)

    

    print('x_img.shape: ',x_img.shape)

    print('x_meta.shape: ',x_meta.shape)

    print('target: {} \n'.format(target))

    

    logits = model(x_img, x_meta)

    print('Logits shape: ', logits.shape)

    print(f'Logits {logits}')

    

    loss = criterion(logits, target)

    print('loss: ',loss)



    print(' \n ----------------------- \n')

'''
df_train_set, df_test_set = train_test_split(df_train, test_size=0.2, random_state=10)
DEBUG = False
# Loading data

if DEBUG:

    train_df_debug = df_train_set.sample(batch_size * 3)

    valid_df_debug = df_test_set.sample(batch_size * 3)

    

    train_Dataset = img_meta_Dataset(train_df_debug, transformer=augmentation)

    valid_Dataset = img_meta_Dataset(valid_df_debug, transformer=augmentation)



    train_DataLoader = DataLoader(train_Dataset, batch_size=3)

    valid_DataLoader = DataLoader(valid_Dataset, batch_size=3)



else:

    train_Dataset = img_meta_Dataset(df_train_set, transformer=augmentation)

    valid_Dataset = img_meta_Dataset(df_test_set, transformer=augmentation)

    

    train_DataLoader = DataLoader(train_Dataset, batch_size=batch_size)

    valid_DataLoader = DataLoader(valid_Dataset, batch_size=batch_size)



# Initiating

model = blood_sweat_tears(out_dim=out_dim)

model = model.to(device)



loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)



train_loss = []

auc_max = 0
# Training

for epoch in range(1, n_epochs + 1):

    print('---------Epoch {}----------'.format(epoch))

    

    PREDS = []

    TARGETS = []

    

    model.train()

    bar = tqdm(train_DataLoader)

    for (x, target) in bar:

        optimizer.zero_grad()

        x_img, x_meta = x

        x_img, x_meta, target = x_img.to(device), x_meta.to(device), target.to(device)  



        logits = model(x_img, x_meta)

        preds = logits.softmax(1)



        loss = criterion(logits, target)

        loss.backward()

        optimizer.step()



        PREDS.append(preds.detach().cpu())

        TARGETS.append(target.detach().cpu())



        # Screen output

        loss_np = loss.detach().cpu().numpy()

        train_loss.append(loss_np)



    PREDS = torch.cat(PREDS).numpy()

    y_pred =  PREDS[:, melanoma_idx]

    

    TARGETS = torch.cat(TARGETS).numpy()

    y_true = (TARGETS==melanoma_idx).astype(float)

    

    acc = (PREDS.argmax(1) == TARGETS).mean() * 100.

    auc = roc_auc_score(y_true, y_pred)

    gini_score = 2 * auc - 1

    

    print('loss: %.2f' % (train_loss[-1]))

    print('Accuracy: {} -- Gini score: {}'.format(acc, gini_score))

    

    best_gini_model_file = f'{kernel_type}_{epoch}_gini_{gini_score}.pth'

    torch.save(model.state_dict(), best_gini_model_file)
# Evaluate WITHOUT TTA

PREDS = []

TARGETS = []

auc_max = 0

gini_max = 0



model.eval()  



# Evaluating

with torch.no_grad():

    bar = tqdm(valid_DataLoader)

    for (x, target) in bar:

        optimizer.zero_grad()

        x_img, x_meta = x

        x_img, x_meta, target = x_img.to(device), x_meta.to(device), target.to(device)  



        logits = model(x_img, x_meta)

        preds = logits.softmax(1)



        loss = criterion(logits, target)



        PREDS.append(preds.detach().cpu())

        TARGETS.append(target.detach().cpu())



        # Screen output

        loss_np = loss.detach().cpu().numpy()

        train_loss.append(loss_np)



# Gini score

PREDS = torch.cat(PREDS).numpy()

TARGETS = torch.cat(TARGETS).numpy()

        

acc = (PREDS.argmax(1) == TARGETS).mean() * 100.

auc = roc_auc_score((TARGETS==melanoma_idx).astype(float), PREDS[:, melanoma_idx])

gini_score = 2 * auc - 1



if gini_score > gini_max:

    print('gini_max ({:.6f} --> {:.6f}). Saving model ...'.format(gini_max, gini_score))

    best_gini_model_file = f'{kernel_type}_{epoch}_gini_{gini_score}.pth'

    torch.save(model.state_dict(), best_gini_model_file)

    gini_max = gini_score



print('Last Gini score: ', gini_score)

print('Best Gini score: {}'.format(gini_max))

  
def TTA_transformer(x_img, n_TTA): # x_img in torch.tensor -- not jpg anymore

    if n_TTA >= 4:

        x_img = x_img.transpose(2,3)

    if n_TTA % 4 == 0:

        return x_img

    elif n_TTA % 4 == 1:

        return x_img.flip(2)

    elif n_TTA % 4 == 2:

        return x_img.flip(3)

    elif n_TTA % 4 == 3:

        return x_img.flip(2).flip(3)

    return x_img
# Evaluate WITH TTA

model.eval()



val_loss = []

PREDICTIONS = []

TARGETS = []



# TTA predictions

with torch.no_grad():

    for (x, target) in tqdm(valid_DataLoader):



        x_img, x_meta = x

        x_img, x_meta, target = x_img.to(device), x_meta.to(device), target.to(device)



        # TTA - Test Time Augmentation

        predictions = torch.zeros((x_img.shape[0], out_dim)).to(device)   

        for n in range(n_TTA):  # TTA - For each data point, predict n_TTA times

            x_img_transformed = TTA_transformer(x_img, n)

            logits = model(x_img_transformed, x_meta)

            preds  = logits.softmax(1)  

        predictions /= n_TTA  # TTA - final result = mean of predictions

        

        loss = criterion(logits, target)

        loss_np = loss.detach().cpu().numpy()

        val_loss.append(loss_np)

        

        PREDICTIONS.append(predictions.detach().cpu())

        TARGETS.append(target.detach().cpu())     

        

val_loss = np.mean(val_loss)

PREDICTIONS = torch.cat(PREDICTIONS).numpy()

TARGETS = torch.cat(TARGETS).numpy()



auc = roc_auc_score((TARGETS==melanoma_idx).astype(float), PREDICTIONS[:, melanoma_idx])

gini_score = 2 * auc - 1



print('loss: %.5f' % (val_loss))

print('Validation Gini score: {}'.format(gini_max))