# Input



dir_csv = '../input/rsna-intracranial-hemorrhage-detection'





# Parameters



n_classes = 6

n_epochs = 2

batch_size = 64
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch, torch.nn as nn

from torchvision import models, transforms, datasets

import matplotlib.pyplot as plt

import torch.nn.functional as F

import math

import pydicom

from collections import Counter

import tqdm

#from apex import amp

import cv2

import glob

import torch.optim as optim

from albumentations import Compose, ShiftScaleRotate, Resize, CenterCrop, HorizontalFlip, RandomBrightnessContrast

from albumentations.pytorch import ToTensor

from torch.utils.data import Dataset

os.listdir('/kaggle/input/rsna-intracranial-hemorrhage-detection')



# Any results you write to the current directory are saved as output.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
# Import and ajust stage_1_train.csv

input_path = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'

df_train = pd.read_csv(input_path + 'stage_1_train.csv')

#df_train['image_ID'] = df_train['ID'].str[:12]

duplicates_to_remove = [

        1598538, 1598539, 1598540, 1598541, 1598542, 1598543,

        312468,  312469,  312470,  312471,  312472,  312473,

        2708700, 2708701, 2708702, 2708703, 2708704, 2708705,

        3032994, 3032995, 3032996, 3032997, 3032998, 3032999

    ]

    

df_train = df_train.drop(index=duplicates_to_remove)

df_train.set_index('ID', inplace=True)

print(len(df_train))

df_train.head(10)
ds_columns = ['ID',

              'PatientID',

              'Modality',

              'StudyInstance',

              'SeriesInstance',

               # 'PhotoInterpretation',

              'Position0', 'Position1', 'Position2']

              #'Orientation0', 'Orientation1', 'Orientation2', 'Orientation3', 'Orientation4', 'Orientation5',

              #'PixelSpacing0', 'PixelSpacing1']

def extract_dicom_features(ds):

    

    ds_items = [ds.SOPInstanceUID,

                ds.PatientID,

                ds.Modality,

                ds.StudyInstanceUID,

                ds.SeriesInstanceUID,

                #ds.PhotometricInterpretation,

                ds.ImagePositionPatient]

                #ds.ImageOrientationPatient,

                #ds.PixelSpacing]



    line = []

    for item in ds_items:

        if type(item) is pydicom.multival.MultiValue:

            line += [float(x) for x in item]

        else:

            line.append(item)



    return {x:y for x, y in zip(ds_columns, line)}
# transform the dicom features into dataset to load faster

class GetCSV(torch.utils.data.Dataset):

    

    def __init__(self, folder_path, transform=None):

        self.folder_path = folder_path

        self.dcm_files = os.listdir(folder_path)

        self.transform = transform

    

    def __len__(self):

        return len(self.dcm_files)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()



        path = self.folder_path + self.dcm_files[idx]

        ds = pydicom.read_file(path)

        sample = extract_dicom_features(ds)

        

        if self.transform:

            sample = self.transform(sample)



        return sample   

    

# Functions



class IntracranialDataset(Dataset):



    def __init__(self, csv_file, path, labels, transform=None):

        

        self.path = path

        self.data = csv_file

        self.transform = transform

        self.labels = labels



    def __len__(self):

        

        return len(self.data)



    def __getitem__(self, idx):

        ID = self.data.loc[idx, 'ID']

        img_name = os.path.join(self.path, ID + '.png')

        img = cv2.imread(img_name)   

        

        try:

            if self.transform:       

                augmented = self.transform(image=img)

                img = augmented['image']

        except:

            img = torch.zeros([3, 200, 200], dtype=torch.float32)

            

        return {'image': img, 'ID': ID}
# Load the dcm files to csv

train_path = input_path + 'stage_1_train_images/'

train_dataset = GetCSV(train_path)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=False, num_workers=4)

test_path = input_path + 'stage_1_test_images/'

test_dataset = GetCSV(test_path)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False, num_workers=4)



def load_dicom_features(loader):



    df_features = []

    for datas in tqdm.tqdm_notebook(loader):

        df_features.append(pd.DataFrame(datas))



    return pd.concat(df_features).reset_index(drop=True)

    

df_features_train = load_dicom_features(train_loader)

df_features_test = load_dicom_features(test_loader)
# inser labels into dataset

catgs = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']

ids = df_features_train['ID'].values

for catg in catgs:

    ids_catg = [ID + '_' + catg for ID in ids]

    df_features_train[catg] = df_train.loc[ids_catg]['Label'].values

    

df_features_train.head()
# Data loaders

dir_train_img = '../input/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x'

dir_test_img = '../input/rsna-test-stage-1-images-png-224x/stage_1_test_png_224x'



transform_train = Compose([CenterCrop(200, 200), ToTensor()])



train_dataset = IntracranialDataset(csv_file=df_features_train, path=dir_train_img, transform=transform_train, labels=False)

test_dataset = IntracranialDataset(csv_file=df_features_test, path=dir_test_img, transform=transform_train, labels=False)



data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
weight_path = '../input/res-net-trained-ct/model.weights'

# Model



device = torch.device("cuda:0")

model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')

model.fc = torch.nn.Linear(2048, n_classes)

model.load_state_dict(torch.load(weight_path, map_location='cpu'))

model.eval()

model.to(device)
def extract_preds(data_loader):

    catgs_pred = [catg + '_pred' for catg in catgs]

    with torch.no_grad():

        img_embedding = pd.DataFrame()

        for sample in tqdm.tqdm_notebook(data_loader):

            ID = sample['ID']

            image = sample['image'].to(device)

            preds = model(image).cpu().numpy()

            new_df = pd.DataFrame(preds, columns=catgs_pred)

            new_df['ID'] = ID

            img_embedding = pd.concat([img_embedding, new_df])

        return img_embedding



img_embedding_train = extract_preds(data_loader_train)

img_embedding_test = extract_preds(data_loader_test)
df_features_test = pd.merge(df_features_test, img_embedding_test, on='ID')

df_features_train = pd.merge(df_features_train, img_embedding_train, on='ID')

del img_embedding_train, img_embedding_test
df_features_train.reset_index().to_csv('features_train.csv', index=False)

df_features_test.reset_index().to_csv('features_test.csv', index=False)