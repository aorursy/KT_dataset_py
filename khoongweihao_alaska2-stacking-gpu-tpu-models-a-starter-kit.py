!pip install -q efficientnet_pytorch

from efficientnet_pytorch import EfficientNet

from albumentations.pytorch import ToTensor

from albumentations import (

    Compose, HorizontalFlip, CLAHE, HueSaturationValue,

    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize,

    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,

    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,

    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip

)

import os

import torch

import pandas as pd

import numpy as np

import random

import torch.nn as nn

import seaborn as sns

import matplotlib.pyplot as plt

from glob import glob

import torchvision

from torch.utils.data import Dataset

import time

from tqdm.notebook import tqdm

# from tqdm import tqdm

from sklearn import metrics

import cv2

import gc

import torch.nn.functional as F
seed = 42

print(f'setting everything to seed {seed}')

random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
data_dir = '../input/alaska2-image-steganalysis'

folder_names = ['JMiPOD/', 'JUNIWARD/', 'UERD/']

class_names = ['Normal', 'JMiPOD_75', 'JMiPOD_90', 'JMiPOD_95', 

               'JUNIWARD_75', 'JUNIWARD_90', 'JUNIWARD_95',

                'UERD_75', 'UERD_90', 'UERD_95']

class_labels = { name: i for i, name in enumerate(class_names)}
train_df = pd.read_csv('../input/alaska2trainvalsplit/alaska2_train_df.csv')

val_df = pd.read_csv('../input/alaska2trainvalsplit/alaska2_val_df.csv')
class Alaska2Dataset(Dataset):



    def __init__(self, df, augmentations=None):



        self.data = df

        self.augment = augmentations



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        fn, label = self.data.loc[idx]

        im = cv2.imread(fn)[:, :, ::-1]

        if self.augment:

            # Apply transformations

            im = self.augment(image=im)

        return im, label





img_size = 512

AUGMENTATIONS_TRAIN = Compose([

    VerticalFlip(p=0.5),

    HorizontalFlip(p=0.5),

    ToFloat(max_value=255),

    ToTensor()

], p=1)





AUGMENTATIONS_TEST = Compose([

    ToFloat(max_value=255),

    ToTensor()

], p=1)

class Net(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        # 1280 is the number of neurons in last layer. is diff for diff. architecture

        self.dense_output = nn.Linear(1280, num_classes)



    def forward(self, x):

        feat = self.model.extract_features(x)

        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)

        return self.dense_output(feat)
!ls ../input/alaska2-efficientnet-trained-model-weights
models = os.listdir('../input/alaska2-efficientnet-trained-model-weights/')

models 
tpu_models = [fn for fn in models if 'model_effnet' in fn]

tpu_models
gpu_models =  [fn for fn in models if 'efficientnetb0_lb' in fn]

gpu_models
# we will not load these here as the TPU weights provided do not help in the stacking
loaded_gpu_models = []

for fn in gpu_models:

    device = 'cuda'

    model = Net(num_classes=len(class_labels)).to(device)

    model.load_state_dict(torch.load(('../input/alaska2-efficientnet-trained-model-weights/' + fn)))

    loaded_gpu_models.append(model)
# # Inference

class Alaska2TestDataset(Dataset):



    def __init__(self, df, augmentations=None):



        self.data = df

        self.augment = augmentations



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        fn = self.data.loc[idx][0]

        im = cv2.imread(fn)[:, :, ::-1]



        if self.augment:

            # Apply transformations

            im = self.augment(image=im)



        return im





test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))

test_df = pd.DataFrame({'ImageFileName': list(

    test_filenames)}, columns=['ImageFileName'])



batch_size = 16

num_workers = 4

test_dataset = Alaska2TestDataset(test_df, augmentations=AUGMENTATIONS_TEST)

test_loader = torch.utils.data.DataLoader(test_dataset,

                                          batch_size=batch_size,

                                          num_workers=num_workers,

                                          shuffle=False,

                                          drop_last=False)
test_df['Id'] = test_df['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])



for k, model in enumerate(loaded_gpu_models):

    model.eval()



    preds = []

    tk0 = tqdm(test_loader)

    with torch.no_grad():

        for i, im in enumerate(tk0):

            inputs = im["image"].to(device)

            # flip vertical

            im = inputs.flip(2)

            outputs = model(im)

            # fliplr

            im = inputs.flip(3)

            outputs = (0.25*outputs + 0.25*model(im))

            outputs = (outputs + 0.5*model(inputs))        

            preds.extend(F.softmax(outputs, 1).cpu().numpy())



    preds = np.array(preds)

    labels = preds.argmax(1)

    new_preds = np.zeros((len(preds),))

    new_preds[labels != 0] = preds[labels != 0, 1:].sum(1)

    new_preds[labels == 0] = 1 - preds[labels == 0, 0]

    

    test_df['Label' + str(k)] = new_preds



test_df = test_df.drop('ImageFileName', axis=1)
test_df.head()
ncol = test_df.shape[1]

test_df.iloc[:,1:ncol].corr()
corr = test_df.iloc[:,1:10].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
test_df['max'] = test_df.iloc[:, 1:ncol].max(axis=1)

test_df['min'] = test_df.iloc[:, 1:ncol].min(axis=1)

test_df['mean'] = test_df.iloc[:, 1:ncol].mean(axis=1)

test_df['median'] = test_df.iloc[:, 1:ncol].median(axis=1)
cutoff_lo = 0.3

cutoff_hi = 0.7
test_df['Label'] = test_df['mean']

test_df[['Id', 'Label']].to_csv('stack_mean.csv', index=False, float_format='%.6f')

test_df['Label'] = test_df['median']

test_df[['Id', 'Label']].to_csv('stack_median.csv', index=False, float_format='%.6f')
test_df['Label'] = np.where(np.all(test_df.iloc[:,1:ncol] > cutoff_lo, axis=1), 1, 

                                    np.where(np.all(test_df.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             0, test_df['median']))

test_df[['Id', 'Label']].to_csv('stack_pushout_median.csv', index=False, float_format='%.6f')
test_df['Label'] = np.where(np.all(test_df.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    test_df['max'], 

                                    np.where(np.all(test_df.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             test_df['min'], 

                                             test_df['mean']))

test_df[['Id', 'Label']].to_csv('stack_minmax_mean.csv', index=False, float_format='%.6f')
test_df['Label'] = np.where(np.all(test_df.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    test_df['max'], 

                                    np.where(np.all(test_df.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             test_df['min'], 

                                             test_df['median']))

test_df[['Id', 'Label']].to_csv('stack_minmax_median.csv',  index=False, float_format='%.6f')