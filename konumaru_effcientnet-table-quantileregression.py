import sys

sys.path = [

    '/kaggle/input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',

] + sys.path
import os

import sys

import cv2

import time

import random

import pickle

import inspect

import datetime

from PIL import Image

from tqdm import tqdm

from copy import deepcopy

from importlib import machinery





import seaborn as sns

sns.set_style("darkgrid")

import matplotlib.pyplot as plt

%matplotlib inline





import numpy as np

import pandas as pd



import torch

import torch.nn as nn

from torch import optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms



from efficientnet_pytorch import EfficientNet



from sklearn import model_selection

from sklearn.preprocessing import StandardScaler, MinMaxScaler



import warnings

warnings.simplefilter('ignore')
if os.path.exists('/kaggle/input'):

    DATA_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression/'

    PROCESSING_DIR = '/kaggle/input/osic-preprocessing-data/'

    IM_FOLDER = '/kaggle/input/osic-average-images/'

else:

    DATA_DIR = '../data/raw/'

    PROCESSING_DIR = '../data/processing/'

    IM_FOLDER = '../data/processing/average_image/'



SEED = 55



NUM_FOLD = 5

IMG_SIZE = 256

BATCH_SIZE = 64

NUM_EPOCH = 20



EFF_NET = 1



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def seed_everything(seed=1234):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(SEED)
def preprocessing(data: pd.DataFrame, is_test: bool = True) -> pd.DataFrame:

    # Create Common Features.

    features = pd.DataFrame()

    for patient, u_data in data.groupby('Patient'):

        feature = pd.DataFrame({

            'current_FVC': u_data['FVC'],

            'current_Percent': u_data['Percent'],

            'current_Age': u_data['Age'],

            'current_Week': u_data['Weeks'],

            'Patient': u_data['Patient'],

            'Sex': u_data['Sex'].map({'Female': 0, 'Male': 1}),

            'SmokingStatus': u_data['SmokingStatus']

#             'SmokingStatus': u_data['SmokingStatus'].map({'Currently smokes': 0, 'Never smoked': 1, 'Ex-smoker': 2}),

        })

        features = pd.concat([features, feature])

    features = pd.get_dummies(features, columns=['SmokingStatus'])

    # Create Label Data.

    if is_test:

        label = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), usecols=['Patient_Week'])

        label['Patient'] = label['Patient_Week'].apply(lambda x: x.split('_')[0])

        label['pred_Weeks'] = label['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)

        label['FVC'] = np.nan



        dst_data = pd.merge(label, features, how='left', on='Patient')

    else:

        label = pd.DataFrame({

            'Patient_Week': data['Patient'].astype(str) + '_' + data['Weeks'].astype(str),

            'Patient': data['Patient'],

            'pred_Weeks': data['Weeks'],

            'FVC': data['FVC']

        })



        dst_data = pd.merge(label, features, how='outer', on='Patient')

        dst_data = dst_data.query('pred_Weeks!=current_Week')



    dst_data = dst_data.assign(passed_Weeks=dst_data['pred_Weeks'] - dst_data['current_Week'])

    return dst_data



# Train Data.

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

train = preprocessing(train, is_test=False)

train.drop(['SmokingStatus_Currently smokes'], axis=1, inplace=True)



train_PiXelStats_path = os.path.join(PROCESSING_DIR, 'train_pixel_stats.csv')

train_picxel_stats = pd.read_csv(train_PiXelStats_path)

train = train.merge(train_picxel_stats, how='left', on='Patient')

train.dropna(axis=0, inplace=True)



# Test Data.

test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

test = preprocessing(test, is_test=True)



test_PiXelStats_path = os.path.join(PROCESSING_DIR, 'test_pixel_stats.csv')

test_picxel_stats = pd.read_csv(test_PiXelStats_path)

test = test.merge(test_picxel_stats, how='left', on='Patient')
print(train.shape)

display(train.head())



print(test.shape)

display(test.head())
# Ref: https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet



class OSICDataset(Dataset):

    

    def __init__(self, df: pd.DataFrame, target: str, imfolder: str, train: bool = True, transforms = None, meta_features: list = None):

        """

        OSIC Dataset for pytorch. 

        

        Parameters

        ----------

        df : pd.Dataframe

            DataFrame with data description.

        target : str

            target column.

        imfolder : str

            folder with images.

        train : bool

            flag  of whether train or test dataset.

        transformers : torchvision.transforms

            image transformation method to be applid

        meta_features : list

            list of features with meta information, such as sex and age.

        """

        self.df = df

        self.target = target

        self.imfolder = imfolder

        self.transforms = transforms

        self.train = train

        self.meta_features = meta_features

        

    def __getitem__(self, index):

        patient = self.df.iloc[index]['Patient']

        im_path = os.path.join(self.imfolder, patient + '.pt')

        

        x = torch.load(im_path)

        meta = self.df.iloc[index][self.meta_features].to_numpy(dtype=np.float32)

        

        if self.transforms:

            x = self.transforms(x)

            

        if self.train:

            y = self.df.iloc[index][self.target]

            return (x, meta), y

        else:

            return (x, meta)

        

    def __len__(self):

        return len(self.df)

        

    

class Net(nn.Module):

    def __init__(self, arch, n_meta_features: int):

        super(Net, self).__init__()

        self.arch = arch

        self.arch._fc = nn.Linear(in_features=1280, out_features=512, bias=True)

        self.meta = nn.Sequential(

            nn.Linear(n_meta_features, 100),

            nn.BatchNorm1d(100),

            nn.ReLU(),

            nn.Dropout(p=0.2),

            nn.Linear(100, 50),  # FC layer output will have 250 features

            nn.BatchNorm1d(50),

            nn.ReLU(),

            nn.Dropout(p=0.2)

        )

        self.ouput = nn.Linear(512 + 50, 3)

        

    def forward(self, inputs):

        x, meta = inputs

        cnn_features = self.arch(x)

        meta_features = self.meta(meta)

        features = torch.cat((cnn_features, meta_features), dim=1)

        output = self.ouput(features)

        return output





def osic_loss(target, pred, sigma):   

    n_sqrt = torch.sqrt(torch.tensor(2.0))

    delta = torch.abs(target - pred)



    sigma[sigma<70] = 70.0

    delta[delta>1000] = 1000.0

    

    metric = - (n_sqrt * delta / sigma) - torch.log(n_sqrt * sigma)

    loss = torch.mean(metric)

    return loss

    



def quantile_loss(preds, target, quantiles):

    assert not target.requires_grad

    assert preds.size(0) == target.size(0)

    losses = []

    for i, q in enumerate(quantiles):

        errors = target - preds[:, i]

        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

    return loss
class CosineAnnealingWarmUpRestarts(_LRScheduler):

    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):

        if T_0 <= 0 or not isinstance(T_0, int):

            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))

        if T_mult < 1 or not isinstance(T_mult, int):

            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))

        if T_up < 0 or not isinstance(T_up, int):

            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))

        self.T_0 = T_0

        self.T_mult = T_mult

        self.base_eta_max = eta_max

        self.eta_max = eta_max

        self.T_up = T_up

        self.T_i = T_0

        self.gamma = gamma

        self.cycle = 0

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

        self.T_cur = last_epoch

    

    def get_lr(self):

        if self.T_cur == -1:

            return self.base_lrs

        elif self.T_cur < self.T_up:

            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]

        else:

            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2

                    for base_lr in self.base_lrs]



    def step(self, epoch=None):

        if epoch is None:

            epoch = self.last_epoch + 1

            self.T_cur = self.T_cur + 1

            if self.T_cur >= self.T_i:

                self.cycle += 1

                self.T_cur = self.T_cur - self.T_i

                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up

        else:

            if epoch >= self.T_0:

                if self.T_mult == 1:

                    self.T_cur = epoch % self.T_0

                    self.cycle = epoch // self.T_0

                else:

                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))

                    self.cycle = n

                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)

                    self.T_i = self.T_0 * self.T_mult ** (n)

            else:

                self.T_i = self.T_0

                self.T_cur = epoch

                

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)

        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):

            param_group['lr'] = lr
class Resize:

    def __init__(self, image_size):

        self.size = image_size

    

    def __call__(self, image):

        image = cv2.resize(image, (self.size, self.size))

        return image



transform = transforms.Compose([

    Resize(IMG_SIZE),

    transforms.ToTensor(),

    transforms.Normalize([0.5], [0.5]),

    transforms.Lambda(lambda x: torch.cat([x, x, x], 0))

])
def plot_metric_loss(history: dict, fold: int):

    """

    Parameters

    ----------

    history : dict

        history keys is train_metric, valid_metric, train_loss, valid_loss.

    epochs : int

        number of fold.

    """

    data_size = len(history['train_metric'])

    

    plt.figure(figsize=(15,5))

    # train and valid metric line

    plt.plot(np.arange(data_size), history['train_metric'], '-o', label='Train Metric', color='#ff7f0e')

    plt.plot(np.arange(data_size), history['valid_metric'], '-o', label='Valid Metric', color='#1f77b4')

    # point best metric epoch

    x = np.argmax(history['valid_metric'])

    y = np.max(history['valid_metric'])

    xdist = plt.xlim()[1] - plt.xlim()[0]

    ydist = plt.ylim()[1] - plt.ylim()[0]

    plt.scatter(x,y,s=200,color='#1f77b4')

    plt.text(x-0.03*xdist, y-0.13*ydist, f'max valid_metric\n{y:.2f}', size=14)

    # Set Label

    plt.ylabel('Metric',size=14)

    plt.xlabel('Epoch',size=14)

    plt.legend(loc=2)



    plt2 = plt.gca().twinx()

    # train and valid loss line

    plt2.plot(np.arange(data_size), history['train_loss'], '-o', label='Train Loss', color='#2ca02c')

    plt2.plot(np.arange(data_size), history['valid_loss'], '-o', label='Valid Loss', color='#d62728')

    # point best loss epoch

    x = np.argmin(history['valid_loss'])

    y = np.min(history['valid_loss'])

    ydist = plt.ylim()[1] - plt.ylim()[0]

    plt.scatter(x, y, s=200, color='#d62728')

    plt.text(x-0.03*xdist, y+0.05*ydist, 'min loss', size=14)

    plt.ylabel('Loss', size=14)

    plt.title(f'FOLD {fold} - Image Size {IMG_SIZE}')

    plt.legend(loc=3)

    plt.show()



# MEMO: plot test

# history = {name: [] for name in ['train_metric', 'valid_metric', 'train_loss', 'valid_loss']}



# history['train_metric'] = np.arange(NUM_EPOCH)

# history['valid_metric'] = np.arange(NUM_EPOCH) - 0.5



# history['train_loss'] = np.random.rand(NUM_EPOCH)

# history['valid_loss'] = np.random.rand(NUM_EPOCH)

    

# plot_metric_loss(history, 0)
effcient_pretrained_path = [

    '../input/efficientnet-pytorch/efficientnet-b0-08094119.pth',

    '../input/efficientnet-pytorch/efficientnet-b1-dbc7070a.pth',

    '../input/efficientnet-pytorch/efficientnet-b2-27687264.pth',

    '../input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth',

    '../input/efficientnet-pytorch/efficientnet-b4-e116e8b3.pth',

    '../input/efficientnet-pytorch/efficientnet-b5-586e6cc6.pth',

    '../input/efficientnet-pytorch/efficientnet-b6-c76e70fd.pth',

    '../input/efficientnet-pytorch/efficientnet-b7-dcc49843.pth'

]



if os.path.exists('/kaggle/input'):

    arch = EfficientNet.from_name(f'efficientnet-b{EFF_NET}')

    arch.load_state_dict(torch.load(effcient_pretrained_path[EFF_NET]))

else:

    arch = EfficientNet.from_pretrained(f'efficientnet-b{EFF_NET}')
drop_cols = ['Patient', 'Patient_Week', 'FVC']

meta_features = [c for c in train.columns.tolist() if c not in drop_cols]



scaler = MinMaxScaler()

scaler.fit(train[meta_features])

train.loc[:, meta_features] = scaler.transform(train[meta_features])

test.loc[:, meta_features] = scaler.transform(test[meta_features])



# ===== Group KFold ======

oof = np.zeros((len(train), 3))

gkf = model_selection.GroupShuffleSplit(n_splits=NUM_FOLD, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(gkf.split(X=train, y=train['FVC'], groups=train['Patient'])):

    print('\n' + '#'*20)

    print('#'*5, f' Fold {fold+1}')

    print('#'*20 + '\n')

    print(f'Train Size: {len(train_idx)}')

    print(f'Valid Size: {len(valid_idx)}', '\n')

    # Model, Optimizer

    model = Net(arch=arch, n_meta_features=len(meta_features)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5)

    # Dataset and Dataloader.

    trainset = OSICDataset(

        df=train.iloc[train_idx].reset_index(drop=True),

        target='FVC',

        imfolder=IM_FOLDER,

        train=True,

        transforms=transform,

        meta_features=meta_features

    )

    validset = OSICDataset(

        df=train.iloc[valid_idx].reset_index(drop=True),

        target='FVC',

        imfolder=IM_FOLDER,

        train=True,

        transforms=transform,

        meta_features=meta_features

    )

    train_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    val_loader = DataLoader(dataset=validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Define Fold Initial Variables.

    updated = False

    patience_cnt = 0

    patience_es = 5

    best_metrc = -999

    history = {name: [] for name in ['train_metric', 'valid_metric', 'train_loss', 'valid_loss']}

    for epoch in range(NUM_EPOCH):

        start_time = time.time()

        # Train Loader

        model.train()

        train_preds = torch.zeros((len(train_idx), 3), dtype=torch.float32, device=device)

        for j, (x, y) in enumerate(train_loader):

            x[0] = torch.tensor(x[0], device=device, dtype=torch.float32)

            x[1] = torch.tensor(x[1], device=device, dtype=torch.float32)

            y = torch.tensor(y, device=device, dtype=torch.float32)



            z = model(x)

            loss = quantile_loss(z, y, (0.2, 0.5, 0.8))

            # Zero gradients, perform a backward pass, and update the weights.

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # Store train predict values.

            start_idx = j * train_loader.batch_size

            end_idx = start_idx + x[0].shape[0]

            train_preds[start_idx:end_idx] = z



        target = torch.tensor(trainset.df[trainset.target], device=device, dtype=torch.float32)

        pred = train_preds[:, 1]

        sigma = train_preds[:, 2] - train_preds[:, 0]

        train_metric = osic_loss(target, pred, sigma).to('cpu').item()

        history['train_metric'].append(train_metric)

        

        train_loss = quantile_loss(train_preds, target, (0.2, 0.5, 0.8)).to('cpu').item()

        history['train_loss'].append(train_loss)



        # Valid Loader

        model.eval()

        valid_preds = torch.zeros((len(valid_idx), 3), dtype=torch.float32, device=device)

        with torch.no_grad():  # Do not calculate gradient since we are only predicting

            for j, (x_val, y_val) in enumerate(val_loader):

                x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32).clone().detach()

                x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32).clone().detach()

                y_val = torch.tensor(y_val, device=device, dtype=torch.float32, requires_grad=True).clone().detach()



                start_idx = j * val_loader.batch_size

                end_idx = start_idx + x_val[0].shape[0]

                valid_preds[start_idx:end_idx] = model(x_val)



        target = torch.tensor(validset.df[validset.target], device=device, dtype=torch.float32)

        pred = valid_preds[:, 1]

        sigma = valid_preds[:, 2] - valid_preds[:, 0]

        valid_metric = osic_loss(target, pred, sigma).to('cpu').item()

        history['valid_metric'].append(valid_metric)

        

        valid_loss = quantile_loss(valid_preds, target, (0.2, 0.5, 0.8)).to('cpu').item()

        history['valid_loss'].append(valid_loss)

        

        # Log.

        train_time = str(datetime.timedelta(seconds=time.time() - start_time))[:7]

        print(f'Epoch {epoch:03}: | Train Loss: {train_loss:.02f} | Valid Loss: {valid_loss:.02f} | Train Metric: {train_metric:.02f} | Valid Metric: {valid_metric:.02f} | Training Time: {train_time}')



        # Early Stopping

        if valid_metric > best_metrc:

            updated = True

            # update OOf values.

            oof[valid_idx] = valid_preds.to('cpu').detach().numpy().copy()

            # update best metric.

            best_metrc = valid_metric

            torch.save(model.state_dict(), f'best_model_{fold+1}_fold.pt')

        else:

            patience_cnt += 1

            

        if patience_cnt >= patience_es:

            # If it has never been updated

            if not updated:

                # update OOf values.

                oof[valid_idx] = valid_preds.to('cpu').detach().numpy().copy()

                # update best metric.

                best_metrc = valid_metric

                torch.save(model.state_dict(), f'best_model_{fold+1}_fold.pt')

            

            print(f"Early stopping: Best Valid Metric is {best_metrc:.02f}")

            break



    plot_metric_loss(history, fold + 1)





# Export OOF.

oof_df = pd.DataFrame(oof, columns=['20', '50', '80'])

oof_df.to_csv('oof.csv', index=False)
target = torch.tensor(train['FVC'], device=device, dtype=torch.float32)

pred = torch.tensor(oof[:, 1], device=device, dtype=torch.float32)

sigma = torch.tensor(oof[:, 2] - oof[:, 0], device=device, dtype=torch.float32)

oof_metric = osic_loss(target, pred, sigma)



print(f'OOF Metric Score: {oof_metric:.2f}')
testset = OSICDataset(

    df=test.reset_index(drop=True),

    target='FVC',

    imfolder=IM_FOLDER,

    train=False,

    transforms=transform,

    meta_features=meta_features

)





preds = np.zeros((len(test), 3))

for fold in range(NUM_FOLD):

    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=True, num_workers=2)

    

    model = Net(arch=arch, n_meta_features=len(meta_features)).to(device)

    model.load_state_dict(torch.load(f'/kaggle/working/best_model_{fold+1}_fold.pt'))

    

    model.eval()

    with torch.no_grad():  # Do not calculate gradient since we are only predicting

        for j, x in enumerate(test_loader):

            x[0] = torch.tensor(x[0], device=device, dtype=torch.float32).clone().detach()

            x[1] = torch.tensor(x[1], device=device, dtype=torch.float32).clone().detach()



            start_idx = j * test_loader.batch_size

            end_idx = start_idx + x[0].shape[0]

            preds[start_idx:end_idx] += model(x).to('cpu').detach().numpy().copy()

        

preds = preds / NUM_FOLD
pred_df = pd.DataFrame({

    'Patient_Week': test['Patient_Week'].values,

    'FVC': preds[:, 1],

    'Confidence': preds[:, 2] - preds[:, 0]

})



submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), usecols=['Patient_Week'])

submission = submission.merge(pred_df, how='left', on='Patient_Week')





if os.path.exists('/kaggle/input'):

    submission.to_csv('submission.csv', index=False)



print(submission.shape)

submission.head()