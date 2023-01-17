!pip install efficientnet_pytorch
import pandas as pd

import numpy as np

import torch

from torchvision import models

from pathlib import Path

Path.ls = lambda x: list(x.iterdir())



import cv2 

import pydicom

from tqdm import tqdm

from matplotlib import pyplot as plt

from torchvision import transforms



from torch import nn

from efficientnet_pytorch import EfficientNet

from efficientnet_pytorch.utils import MemoryEfficientSwish



from torch.optim import Adam

from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
class Config:

    def __init__(self):

        self.FOLDS = 2

        self.EPOCHS = 1

        self.DEVICE = 'cuda'

        self.TRAIN_BS = 64

        self.VALID_BS = 128

        self.model_type = 'b3'

        self.loss_fn = nn.L1Loss()

        

config = Config()
path = Path('/kaggle/input/osic-pulmonary-fibrosis-progression/')

path.ls()
train_df = pd.read_csv(path/'train.csv')

train_df.head()
train_df = train_df.drop(np.nonzero(np.array(train_df['Patient'] == 'ID00011637202177653955184',dtype=float))[0], axis=0).reset_index(drop=True)

train_df = train_df.drop(np.nonzero(np.array(train_df['Patient'] == 'ID00052637202186188008618',dtype=float))[0], axis=0).reset_index(drop=True)
def get_tab(df):

    vector = [(df['Weeks'].values[0] - 30 )/30]

    

    if df.Sex.values[0] == 'Male':

        vector.append(0)

    else: 

        vector.append(1)

    

    if df['SmokingStatus'].values[0] == 'Never smoked':

        vector.extend([0,0])

    elif df['SmokingStatus'].values[0] == 'Currently smokes':

        vector.extend([0,1])

    elif df['SmokingStatus'].values[0] == 'Ex-smoker':

        vector.extend([1,0])

    else :

        vector.extend([1,1])

    return np.array(vector)

 
TAB = {}

TARGET = {}

Person = []



for i, p in enumerate(train_df.Patient.unique()):

    sub = train_df.loc[train_df.Patient == p]

    fvc = sub.FVC.values

    weeks = sub.Weeks.values

    c = np.vstack([weeks, np.ones(len(weeks))]).T

    a, b = np.linalg.lstsq(c, fvc)[0]

    

    TARGET[p] = a

    TAB[p] = get_tab(sub)

    Person.append(p)



Person = np.array(Person)
def get_img(path):

    d = pydicom.dcmread(path)

    return cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (512,512))
class Dataset:

    def __init__(self, path, df, tabular, targets, folder = 'train'):

        self.df = df

        self.tabular = tabular

        self.targets = targets

        self.folder = folder

        self.path = path

        self.transform = transforms.Compose([

            transforms.ToTensor()

        ])

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self,idx):

        row = self.df.loc[idx,:]

        pid = row['Patient']

        # Path to record

        record = self.path/self.folder/pid

        # select image id

        try: 

            

            img_id =  np.random.choice(len(record.ls()))

            

            img = get_img(record.ls()[img_id])

            img = self.transform(img)

            tab = torch.from_numpy(self.tabular[pid]).float()

            target = torch.tensor(self.targets[pid])

            

            return (img,tab), target

        except Exception as e:

            print(e)

            print(pid, img_id)
def collate_fn(b):

    xs, ys = zip(*b)

    imgs, tabs = zip(*xs)

    return (torch.stack(imgs).float(),torch.stack(tabs).float()),torch.stack(ys).float()
class Model(nn.Module):

    def __init__(self,eff_name='b0'):

        super().__init__()

        self.input = nn.Conv2d(1,3,kernel_size=3,padding=1,stride=2)

        self.bn = nn.BatchNorm2d(3)

        self.model = EfficientNet.from_pretrained(f'efficientnet-{eff_name}')

        self.model._fc = nn.Linear(1536, 500, bias=True)

        self.meta = nn.Sequential(nn.Linear(4, 500),

                                  nn.BatchNorm1d(500),

                                  nn.ReLU(),

                                  nn.Dropout(p=0.2),

                                  nn.Linear(500,250),

                                  nn.BatchNorm1d(250),

                                  nn.ReLU(),

                                  nn.Dropout(p=0.2))

        self.output = nn.Linear(500+250, 1)

        self.relu = nn.ReLU()

    

    def forward(self, x,tab):

        x = self.relu(self.bn(self.input(x)))

        x = self.model(x)

        tab = self.meta(tab)

        x = torch.cat([x, tab],dim=1)

        return self.output(x)
from sklearn.model_selection import KFold



def get_split_idxs(n_folds=5):

    kv = KFold(n_splits=n_folds)

    splits = []

    for i,(train_idx, valid_idx) in enumerate(kv.split(Person)):

        splits.append((train_idx, valid_idx))

        

    return splits
splits = get_split_idxs(n_folds=config.FOLDS)
def train_loop(model, dl, opt, sched, device, loss_fn):

    model.train()

    for X,y in dl:

        imgs = X[0].to(device)

        tabs = X[1].to(device)

        y = y.to(device)

        outputs = model(imgs, tabs)

        loss = loss_fn(outputs.squeeze(), y)

        opt.zero_grad()

        loss.backward()

        opt.step()

        if sched is not None:

            sched.step()

            



def eval_loop(model, dl, device, loss_fn):

    model.eval()

    final_outputs = []

    final_loss = []

    with torch.no_grad():

        for X,y in dl:

            imgs = X[0].to(device)

            tabs = X[1].to(device)

            y=y.to(device)



            outputs = model(imgs, tabs)

            loss = loss_fn(outputs.squeeze(), y)



            final_outputs.extend(outputs.detach().cpu().numpy().tolist())

            final_loss.append(loss.detach().cpu().numpy())

        

    return final_outputs, final_loss
from functools import partial



def apply_mod(m,f):

    f(m)

    for l in m.children(): apply_mod(l,f)



def set_grad(m,b):

    if isinstance(m, (nn.Linear, nn.BatchNorm2d)): return 

    if hasattr(m, 'weight'):

        for p in m.parameters(): p.requires_grad_(b)



models = {}

for i in range(config.FOLDS):

    models[i] = Model(config.model_type)
for k,v in models.items():

    apply_mod(v.model, partial(set_grad, b=False))
history = []
for i, (train_idx, valid_idx) in enumerate(splits):

    print(f"===================Fold : {i} ================")



    train = train_df.loc[train_df['Patient'].isin(Person[train_idx])].reset_index(drop=True)

    valid = train_df.loc[train_df['Patient'].isin(Person[valid_idx])].reset_index(drop=True)





    train_ds = Dataset(path, train, TAB, TARGET)

    train_dl = torch.utils.data.DataLoader(

        dataset=train_ds,

        batch_size=config.TRAIN_BS,

        shuffle=True,

        collate_fn=collate_fn        

    )



    valid_ds = Dataset(path, valid, TAB, TARGET)

    valid_dl = torch.utils.data.DataLoader(

        dataset=valid_ds,

        batch_size=config.VALID_BS,

        shuffle=False,

        collate_fn=collate_fn

    )



    model = models[i]

    model.to(config.DEVICE)

    lr=1e-3

    momentum = 0.9

    

    num_steps = len(train_dl)

    optimizer = Adam(model.parameters(), lr=lr,weight_decay=0.1)

    scheduler = OneCycleLR(optimizer, 

                           max_lr=lr,

                           epochs=config.EPOCHS,

                           steps_per_epoch=num_steps

                           )

    sched = ReduceLROnPlateau(optimizer,

                              verbose=True,

                              factor=0.1)

    losses = []

    for epoch in range(config.EPOCHS):

        print(f"=================EPOCHS {epoch+1}================")

        train_loop(model, train_dl, optimizer, scheduler, config.DEVICE,config.loss_fn)

        metrics = eval_loop(model, valid_dl,config.DEVICE,config.loss_fn)

        total_loss = np.array(metrics[1]).mean()

        losses.append(total_loss)

        print("Loss ::\t", total_loss)

        sched.step(total_loss)

        

    model.to('cpu')

    history.append(losses)

    

    

        
for k, m in models.items():

    torch.save(m.state_dict(), f'fold_{k}.pth')