!pip install efficientnet_pytorch
import pandas as pd

import numpy as np

from pathlib import Path

Path.ls = lambda x: list(x.iterdir())

import os



from pydicom import dcmread

import cv2



from tqdm import tqdm

from matplotlib import pyplot as plt



import torch

from torch import nn

from torch.optim import Adam

from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from torchvision import transforms



from efficientnet_pytorch import EfficientNet

from efficientnet_pytorch.utils import MemoryEfficientSwish

path = Path('/kaggle/input/osic-pulmonary-fibrosis-progression/')

path.ls()
train_df = pd.read_csv(path/'train.csv')

train_df.head()
err_patients = ['ID00011637202177653955184','ID00052637202186188008618']
train_df = train_df.drop(np.nonzero(np.array(train_df['Patient'] == 'ID00011637202177653955184',dtype=float))[0], axis=0).reset_index(drop=True)

train_df = train_df.drop(np.nonzero(np.array(train_df['Patient'] == 'ID00052637202186188008618',dtype=float))[0], axis=0).reset_index(drop=True)
def get_tab(df):

    vector = [(df['Age'].values[0] - 30 )/30]

    

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
for patient in err_patients:

    if patient in Person:

        Person.remove(patient)
def get_img(path,size = 512):

    d = dcmread(path)

    return cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (size,size))
def get_files(path,p_ids):

    path = Path(path)    

    return _get_files(path,p_ids)
def _get_files(p, ds, extensions=['.dcm']):

    p = Path(p)

    paths = [p/d for d in ds]

    files = [Path(file.path) for path in paths for file in os.scandir(path) if file is not None and Path(file.path).suffix in extensions]

    return files
len(get_files(path/'train',Person.tolist()))
class Dataset:

    def __init__(self,path,tabular,targets,p_ids, size=224, folder='train'):

        self.path = Path(path)

        self.tabular = tabular

        self.targets = targets

        self.folder = folder

        self.size = size

        self.transforms = transforms.Compose([

            transforms.ToTensor()

        ])

        self.p_ids = p_ids

        self.files = get_files(self.path/folder,self.p_ids)

    

    def __len__(self):

        return len(self.files)

    

    def __getitem__(self,idx):

        img_path = self.files[idx]

        pid = img_path.parent.name

        img = get_img(img_path,self.size)

        img = self.transforms(img)

        tab = torch.from_numpy(self.tabular[pid]).float()

        target = torch.tensor(self.targets[pid])

        return (img,tab), target
def collate_fn(b):

    xs, ys = zip(*b)

    imgs, tabs = zip(*xs)

    return (torch.stack(imgs).float(),torch.stack(tabs).float()),torch.stack(ys).float()
data = Dataset(path,TAB,TARGET, Person)
dl = torch.utils.data.DataLoader(data,shuffle=True,batch_size=8,collate_fn=collate_fn)
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
def train_loop(model, dl, opt, sched, device, loss_fn):

    model.train()

    for X,y in tqdm(dl,total=len(dl)):

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

        for X,y in tqdm(dl,total=len(dl)):

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



class Config:

    def __init__(self,lr, epochs = 2, model_type='b3',device='cuda',bs=64):

        self.FOLDS = 5

        self.EPOCHS = epochs

        self.DEVICE = device

        self.TRAIN_BS = bs

        self.VALID_BS = bs*2

        self.model_type = model_type

        self.loss_fn = nn.L1Loss()

        self.lr = lr
class ModelUtils:

    def __init__(self,config):

        self.models = {}

        self.config = config

        

        self.init_models()

        

    def __getattr__(self,name):

        return getattr(self.config, name)

        

    def init_models(self):

        for i in range(self.FOLDS):

            self.models[i] = Model(self.model_type)

    

    def freeze(self):

        for k,v in self.models.items():

            apply_mod(v.model, partial(set_grad, b=False))

    

    def unfreeze(self):

        for k,v in self.models.items():

            apply_mod(v.model, partial(set_grad, b=True))

    

    def save_model(self,model,fold,config):

        torch.save(model.state_dict(),f"eff_{config.model_type}_fold_{fold}.pth")

        
history = {}
def fit(config, utils, save_weights = True):

    for i, (train_idx, valid_idx) in enumerate(splits):

        print(f"===================Fold : {i} ================")



#         train = train_df.loc[train_df['Patient'].isin(Person[train_idx])].reset_index(drop=True)

#         valid = train_df.loc[train_df['Patient'].isin(Person[valid_idx])].reset_index(drop=True)

        train_ids = Person[train_idx]

        valid_ids = Person[valid_idx]



        train_ds = Dataset(path, TAB, TARGET,train_ids,size=224)

        train_dl = torch.utils.data.DataLoader(

            dataset=train_ds,

            batch_size=config.TRAIN_BS,

            shuffle=True,

            collate_fn=collate_fn,

            num_workers = 4

        )



        valid_ds = Dataset(path, TAB, TARGET,valid_ids,size=224)

        valid_dl = torch.utils.data.DataLoader(

            dataset=valid_ds,

            batch_size=config.VALID_BS,

            shuffle=False,

            collate_fn=collate_fn,

            num_workers = 4

        )



        model = utils.models[i]

        model.to(config.DEVICE)

        lr=config.lr

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

                                  factor=0.1,

                                  patience=3                                

                                 )

        losses = []

        best_loss = 999

        for epoch in range(config.EPOCHS):

            print(f"=================EPOCHS {epoch+1}================")

            train_loop(model, train_dl, optimizer, scheduler, config.DEVICE,config.loss_fn)

            metrics = eval_loop(model, valid_dl,config.DEVICE,config.loss_fn)

            mean_loss = np.array(metrics[1]).mean()

            losses.append(mean_loss)

            print("Loss ::\t", mean_loss)

            sched.step(mean_loss)

            if mean_loss < best_loss:

                best_loss = mean_loss

                if save_weights:

                    print('saving')

                    utils.save_model(model, i, config)



        model.to('cpu')

        history[i] = losses

config = Config(lr=1e-3,bs=256,)

model_utils = ModelUtils(config)
splits = get_split_idxs(n_folds=config.FOLDS)
model_utils.freeze()

fit(config, model_utils)
model_utils.save_model(model_utils.models[0],1,config)
model_utils.unfreeze()
config.EPOCHS = 5
fit(config, model_utils)