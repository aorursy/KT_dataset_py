import pydicom
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import random
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2
import glob, os
import re
import matplotlib.pyplot as plt

import gc
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 50
path = Path('/kaggle/input/osic-pulmonary-fibrosis-progression/')
assert path.exists()
TRAIN_TYPES={"Patient": "category", 
         "Weeks": "int16", "FVC": "int32", 'Percent': 'float32', "Age": "uint8",
        "Sex": "category", "SmokingStatus": "category" }
SUBMISSION_TYPES={"Patient_Week": "category", "FVC": "int32", "Confidence": "int16"}


def read_data(path):
    train_df = pd.read_csv(path/'train.csv', dtype = TRAIN_TYPES)
    test_df = pd.read_csv(path/'test.csv', dtype = TRAIN_TYPES)
    submission_df = pd.read_csv(path/'sample_submission.csv', dtype = SUBMISSION_TYPES)
    train_df.drop_duplicates(keep='first', inplace=True, subset=['Patient','Weeks'])
    return train_df, test_df, submission_df
train_df, test_df, submission_df = read_data(path)
def prepare_submission(df, test_df):
    df['Patient'] = df['Patient_Week'].apply(lambda x:x.split('_')[0])
    df['Weeks'] = df['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
    df = df[['Patient','Weeks','Confidence','Patient_Week']]
    df = df.merge(test_df.drop('Weeks', axis=1).copy(), on=['Patient'])
    return df

submission_df = prepare_submission(submission_df, test_df)
train_df['WHERE'] = 'train'
test_df['WHERE'] = 'val'
submission_df['WHERE'] = 'test'
data = train_df.append([test_df, submission_df])
data['min_week'] = data['Weeks']
data.loc[data.WHERE=='test','min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
base = data.loc[data.Weeks == data.min_week]
base = base[['Patient','FVC', 'Percent']].copy()
base.columns = ['Patient','min_FVC', 'min_Percent']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base
data = data.merge(base, on='Patient', how='left')
from sklearn import datasets, linear_model

slope_map = {}
intercept_map = {}
for i, p in tqdm(enumerate(train_df.Patient.unique())):
    sub = train_df.loc[train_df.Patient == p, :] 
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    lin_model = linear_model.Ridge()
    a, b = np.linalg.lstsq(c, fvc, rcond=None)[0]
    lin_model.fit(X = weeks.reshape([-1, 1]), y = fvc)
    slope_map[p] = lin_model.coef_[0]
    intercept_map[p] = np.log(lin_model.intercept_) * -1
BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
def get_img(path):
    d = pydicom.dcmread(path)
    return cv2.resize(d.pixel_array / 2**11, IMG_DIM)
try:
    get_img(f'{path}/train/ID00219637202258203123958/9.dcm').shape
except:
    print('Image not found')
def read_image_slope(ds_type = 'train'):
    img_folders = [t[0] for t in os.walk(path/f"{ds_type}") if re.match(r'.+ID\d+$', t[0])]
    training_data = []
    folder_count = {}
    for f in img_folders:
        patient = re.sub(r'.+/(.+)', r'\1', f)
        if patient not in BAD_ID:
            slope = slope_map[patient]
            ldir = glob.glob(str(f'{f}/*.dcm'))
            folder_count[patient] = len(ldir)
            for img_file in ldir:
                try:
                    file_name = re.sub(r'.+/(.+)', r'\1', img_file)
                    if re.match(r'\d+\..*', file_name):
                        if int(file_name[:-4]) / len(ldir) < 0.8 and int(file_name[:-4]) / len(ldir) > 0.15:
                            training_data.append((img_file, slope))
                except:
                    print(f'Failed on f{img_file}')
    return training_data, folder_count
img_slope_data, folder_count = read_image_slope()
IMG_DIM = (256, 256)
from sklearn.model_selection import train_test_split 

train_image_slope, val_image_slope = train_test_split(img_slope_data, shuffle=True, train_size= 0.9) 
def get_mean_std(ds):
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data, _ in ds:
        channels_sum += torch.mean(data, dim=[-2, -1])
        channels_squared_sum += torch.mean(data**2, dim=[-2, -1])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std
%%time

# mean,std = get_mean_std(CombinedImageDataset(train_image_slope))
mean = torch.tensor([-0.0149])
std = torch.tensor([0.5000])
class CombinedImageDataset(Dataset):
    def __init__(self, img_slope_data, transforms=transforms.Compose([transforms.ToTensor()])):
        self.data = img_slope_data
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        path, slope_intercept = self.data[i]
        slope= slope_intercept
        image = get_img(path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image.float(), slope
    
    def get_patient(self, i):
        path = self.data[i][0]
        return re.sub(r'.+/(ID.+?)/.+', r'\1', path)
        
    def __repr__(self):
        return  f'patients: {len(self.data)}, image_path: {self.image_path}, transforms: {self.transforms}'
train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
train_ds = CombinedImageDataset(train_image_slope, transforms=train_transform)
len(train_ds)
val_ds = CombinedImageDataset(val_image_slope, transforms=test_transform)
len(train_ds)
BATCH_SIZE = 64
NUM_WORKERS = 4

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def move_to_dev(items):
    return [i.to(device) for i in items]
def freeze(model, layers=7, requires_grad=False):
    ct = 0
    for name, child in model.named_children():
        ct += 1
        if ct < layers:
            for _, params in child.named_parameters():
                params.requires_grad = False
def unfreeze_all(model):
    # Unfreeze model weights
    for param in model.parameters():
        param.requires_grad = True
def create_model():
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(torch.load('/kaggle/input/resnet50/resnet50.pth'))
    num_ftrs = model.fc.in_features
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(num_ftrs, 1)
    freeze(model)
    model = model.to(device);
    return model
model1 = create_model()
sample_img, sample_slope = move_to_dev(next(iter(train_dl)))
sample_img.shape, sample_slope
sample_out = model1(sample_img)
slope_mse = torch.mean((sample_out[:,0] - sample_slope) ** 2)
slope_mse
LR=1e-4
criterion = nn.MSELoss()
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def eval_loop(valid_dl, model):
    with torch.no_grad():
        model.eval()
        total_eval_loss = 0
        total_eval_score = 0
        for val_vals in valid_dl:
            x, y_slope = move_to_dev(val_vals)
            output = model(x)
            loss = criterion(y_slope.unsqueeze(-1), output)
            total_eval_loss += loss.item()
            total_eval_score += loss.item()

        avg_val_loss = total_eval_loss / len(valid_dl)
        avg_val_score = total_eval_score / len(valid_dl)
        return {
            'avg_val_loss': avg_val_loss,
            'avg_val_score': avg_val_score
        }
def train_loop(epochs, train_dl, valid_dl, model, lr = 1e-3, print_score=False, model_name='test'):
    steps = len(train_dl) * epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dl), epochs=epochs)
    avg_train_losses = []
    avg_val_losses = []
    avg_val_scores = []
    lr = []
    best_avg_val_score = 1000
    for epoch in tqdm(range(epochs), total=epochs):
        model.train()
        total_train_loss = 0.0
        t = tqdm(enumerate(train_dl), total=len(train_dl))
        for i, train_vals in t:
            x, y_slope = move_to_dev(train_vals)
            model.zero_grad()
            output = model(x)
            loss = criterion(y_slope.unsqueeze(-1), output)
            t.set_postfix({'loss': loss.item()})
            total_train_loss += loss.item()
            
            # Backward Pass and Optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr.append(get_lr(optimizer))
        
        avg_train_loss = total_train_loss / len(train_dl)
        avg_train_losses.append(avg_train_loss)
        eval_res = eval_loop(valid_dl, model)
        avg_val_loss = eval_res['avg_val_loss']
        avg_val_score = eval_res['avg_val_score']
        avg_val_losses.append(avg_val_loss)
        avg_val_scores.append(avg_val_score)
        if best_avg_val_score > avg_val_score:
            best_avg_val_score = avg_val_score
            # save best model
            print(f'Best model: {best_avg_val_score}')
#             torch.save(model.state_dict(), model_path/f'best_model_images_{model_name}.pt')
        if print_score:
            print(f'{epoch}: avg_val_score: {avg_val_score}')
    return pd.DataFrame({'avg_train_losses': avg_train_losses, 'avg_val_losses': avg_val_losses, 'avg_val_scores': avg_val_scores}), pd.DataFrame({'lr': lr})
NUM_EPOCHS = 2
res_df, lr_df = train_loop(NUM_EPOCHS, train_dl, val_dl, model1, lr=LR, print_score=True)
res_df[['avg_train_losses', 'avg_val_losses']].plot()
UNFROZEN_EPOCHS=2
unfreeze_all(model1)
res_df, lr_df = train_loop(UNFROZEN_EPOCHS, train_dl, val_dl, model1, lr=LR / 10, print_score=True)
res_df[['avg_train_losses', 'avg_val_losses']].plot()
test_orig_df = pd.read_csv(path/'test.csv', dtype = TRAIN_TYPES)
sub_df = pd.read_csv(path/'sample_submission.csv', dtype = SUBMISSION_TYPES)
current_folder = 'test'

def load_data(current_df, current_folder, ldir, p, x):
    for i in ldir:
        if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:
            x.append(get_img(f'/kaggle/input/osic-pulmonary-fibrosis-progression/{current_folder}/{p}/{i}'))


quantiles = [0.2, 0.5, 0.8]
unique_patients = test_orig_df.Patient.unique()

with torch.no_grad():
    model1.eval()
    A_test, B_test, P_test, WEEK = {}, {}, {},{}
    for p in unique_patients:
        x = []
        ldir = os.listdir(f'/kaggle/input/osic-pulmonary-fibrosis-progression/{current_folder}/{p}/')
        load_data(test_orig_df, current_folder, ldir, p, x)
        if len(x) <= 1:
            continue

        x = np.expand_dims(x, axis=-1)
        _a = model1(torch.tensor(x).squeeze().unsqueeze(1).float().to(device))

        # A = slopes, B = intercepts
        A_test[p] = [np.quantile(_a.cpu(), q) for q in quantiles]
        B_test[p] = test_orig_df.FVC.values[test_orig_df.Patient == p] - A_test[p] * test_orig_df.Weeks.values[test_orig_df.Patient == p]
        P_test[p] = test_orig_df.Percent.values[test_orig_df.Patient == p]
        WEEK[p] = test_orig_df.Weeks.values[test_orig_df.Patient == p]

def fvc_calc(a, x, b): return a * x + b
        
for k in sub_df.Patient_Week.values:
    p, w = k.split('_')
    w = int(w)

    fvc = fvc_calc(A_test[p][1], w, B_test[p][1])
    sub_df.loc[sub_df.Patient_Week == k, 'FVC'] = fvc
    conf_1 = np.abs(P_test[p] - A_test[p][1] * abs(WEEK[p] - w))
    conf_2 = np.abs(fvc_calc(A_test[p][2], w, B_test[p][2]) - fvc_calc(A_test[p][0], w, B_test[p][0]))
    sub_df.loc[sub_df.Patient_Week == k, 'Confidence'] = np.clip(np.average([conf_1, conf_2], axis=0), 100, 1000)

sub_df
sub_df.describe().T
sub_df[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)
submission_final_df = pd.read_csv("submission.csv")
for p in test_df['Patient'].unique():
    submission_final_df[submission_final_df['Patient_Week'].str.find(p) == 0]['FVC'].plot()
