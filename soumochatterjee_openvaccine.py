!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
!pip install torchcontrib
import warnings
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import pandas as pd

df = pd.read_json ('../input/stanford-covid-vaccine/train.json', lines=True)

df.head()
import os.path
from os import path

path.exists("../input/stanford-covid-vaccine/bpps/id_724185d34.npy")
test_df = pd.read_json ('../input/stanford-covid-vaccine/test.json', lines=True)

test_df.head()
df = df.drop(['index'], axis=1)
df.head()
df.info()
import statistics
round(statistics.mean(df.reactivity_error[0]) , 4)
df['reactivity_error'] = df['reactivity_error'].apply(lambda x : round(statistics.mean(x) , 4))
df.head()
df['deg_error_Mg_pH10'] = df['deg_error_Mg_pH10'].apply(lambda x : round(statistics.mean(x) , 4))
df['deg_error_pH10'] = df['deg_error_pH10'].apply(lambda x : round(statistics.mean(x) , 4))
df['deg_error_Mg_50C'] = df['deg_error_Mg_50C'].apply(lambda x : round(statistics.mean(x) , 4))
df['deg_error_50C'] = df['deg_error_50C'].apply(lambda x : round(statistics.mean(x) , 4))
df['reactivity'] = df['reactivity'].apply(lambda x : round(statistics.mean(x) , 4))
df['deg_Mg_pH10'] = df['deg_Mg_pH10'].apply(lambda x : round(statistics.mean(x) , 4))
df['deg_pH10'] = df['deg_pH10'].apply(lambda x : round(statistics.mean(x) , 4))
df['deg_Mg_50C'] = df['deg_Mg_50C'].apply(lambda x : round(statistics.mean(x) , 4))
df['deg_50C'] = df['deg_50C'].apply(lambda x : round(statistics.mean(x) , 4))

df.head()
df.describe()
df.sequence[0].count('G')
df['G_in_sequence'] = df['sequence'].apply(lambda x : x.count('G'))
df.head()
set("".join(list(df["predicted_loop_type"])))
df['A_in_sequence'] = df['sequence'].apply(lambda x : x.count('A'))
df['U_in_sequence'] = df['sequence'].apply(lambda x : x.count('U'))
df['C_in_sequence'] = df['sequence'].apply(lambda x : x.count('C'))

df['._in_structure'] = df['structure'].apply(lambda x : x.count('.'))
df['(_in_structure'] = df['structure'].apply(lambda x : x.count('('))
df[')_in_structure'] = df['structure'].apply(lambda x : x.count(')'))

df['B_in_predicted_loop_type'] = df['predicted_loop_type'].apply(lambda x : x.count('B'))
df['E_in_predicted_loop_type'] = df['predicted_loop_type'].apply(lambda x : x.count('E'))
df['H_in_predicted_loop_type'] = df['predicted_loop_type'].apply(lambda x : x.count('H'))
df['I_in_predicted_loop_type'] = df['predicted_loop_type'].apply(lambda x : x.count('I'))
df['M_in_predicted_loop_type'] = df['predicted_loop_type'].apply(lambda x : x.count('M'))
df['S_in_predicted_loop_type'] = df['predicted_loop_type'].apply(lambda x : x.count('X'))
df['X_in_predicted_loop_type'] = df['predicted_loop_type'].apply(lambda x : x.count('S'))

df = df.drop(['sequence', 'structure', 'predicted_loop_type' , 'seq_length' , 'seq_scored'], axis=1)

df.head()
df.columns
submission = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

submission.head()
colums_order = ['id','G_in_sequence','A_in_sequence', 'U_in_sequence', 'C_in_sequence',
        '._in_structure','(_in_structure', ')_in_structure',
        'B_in_predicted_loop_type','E_in_predicted_loop_type', 'H_in_predicted_loop_type',
       'I_in_predicted_loop_type', 'M_in_predicted_loop_type',
       'S_in_predicted_loop_type', 'X_in_predicted_loop_type',
        'signal_to_noise', 'SN_filter', 'reactivity_error', 'deg_error_Mg_pH10',
       'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C',
        'reactivity','deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
df = df[colums_order]

for i in range(1,-5):
    df[colums_order[i]] = df[colums_order[i]].apply(lambda x : round(float(x),4))

df.head()
df.describe()
import numpy as np 
import matplotlib.pyplot as plt

data = np.load('../input/stanford-covid-vaccine/bpps/id_0051b1d76.npy')
print(data.shape)
plt.imshow(data)
plt.show()
import cv2
img = np.load('../input/stanford-covid-vaccine/bpps/id_000ae4237.npy') # ../input/stanford-covid-vaccine/bpps/id_09be4ee60.npy
        
img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

print(img.shape)
plt.imshow(img)
plt.show()
from torch.utils.data import Dataset
import cv2
import torch
from torchvision import transforms
import albumentations
from PIL import Image

class openVaccine(Dataset):
    def __init__(self, id , tabular , image, mean , std , is_valid):
        self.id = id
        self.tabular = tabular
        self.image = image
        self.is_valid = is_valid
        if self.is_valid == 1: # transforms for validation images
            self.aug = albumentations.Compose([
               albumentations.Normalize(mean , std , always_apply = True) 
            ])
        else:                  # transfoms for training images 
            self.aug = albumentations.Compose([
                albumentations.Normalize(mean , std , always_apply = True),
                albumentations.ShiftScaleRotate(shift_limit = 0.0625,
                                                scale_limit = 0.1 ,
                                                rotate_limit = 5,
                                                p = 0.9)
            ]) 
            
            
        self.reactivity = tabular.reactivity.values
        self.deg_Mg_pH10 = tabular.deg_Mg_pH10.values
        self.deg_pH10 = tabular.deg_pH10.values
        self.deg_Mg_50C = tabular.deg_Mg_50C.values
        self.deg_50C = tabular.deg_50C.values
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, index):
        id = self.id[index]
        
        # converting jpg format of images to numpy array
        img = np.load('../input/stanford-covid-vaccine/bpps/'+ self.image[index] +'.npy') 
        
        img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(img).convert('RGB')
        img = self.aug(image = np.array(img))['image']
        img = np.transpose(img, (2,0,1)).astype(np.float32) # 2,0,1 because pytorch excepts image channel first then dimension of image
        
        tabular = self.tabular.iloc[:,:]
        
        X = tabular[['G_in_sequence','A_in_sequence', 'U_in_sequence', 'C_in_sequence',
                     '._in_structure','(_in_structure', ')_in_structure',
                     'B_in_predicted_loop_type','E_in_predicted_loop_type', 'H_in_predicted_loop_type','I_in_predicted_loop_type', 'M_in_predicted_loop_type','S_in_predicted_loop_type', 'X_in_predicted_loop_type',
                     'signal_to_noise', 'SN_filter', 
                     'reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C']]
        X = X.values[index]
        
       
        return {
            'image' : torch.tensor(img, dtype = torch.long) , 
            'tabular_data' : torch.tensor(X, dtype = torch.float) , 
            'reactivity_output' : torch.tensor(self.reactivity[index], dtype = torch.float), 
            'deg_Mg_pH10_output' : torch.tensor(self.deg_Mg_pH10[index], dtype = torch.float), 
            'deg_pH10_output' : torch.tensor(self.deg_pH10[index], dtype = torch.float),  
            'deg_Mg_50C_output' : torch.tensor(self.deg_Mg_50C[index], dtype = torch.float),  
            'deg_50C_output' : torch.tensor(self.deg_50C[index], dtype = torch.float)
        }
# split the data into train and test set
from sklearn import model_selection
df_train, df_valid = model_selection.train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
df_train = df_train.reset_index(drop = True)
df_valid = df_valid.reset_index(drop = True)
df_train.head()
df_valid.head()
len(df_train)
# prepare transforms standard to MNIST
train_data = openVaccine(id = [i for i in range(len(df_train))], 
                         tabular = df_train, 
                         image = df_train['id'],  
                         mean = (0.485, 0.456, 0.406),
                         std = (0.229, 0.224, 0.225) , is_valid = 0)

val_data = openVaccine(id = [i for i in range(len(df_valid))], 
                       tabular = df_valid, 
                       image = df_valid['id'],  
                       mean = (0.485, 0.456, 0.406),
                       std = (0.229, 0.224, 0.225) , is_valid = 1)
#dry run 
idx = 100 # taking validation data index for 100th image out of 51000 images

img = val_data[idx]["image"]
plt.imshow(np.transpose(img, (1,2,0)))
print(val_data[idx]["tabular_data"])
print(val_data[idx]["reactivity_output"])
print(val_data[idx]["deg_Mg_pH10_output"])
print(val_data[idx]["deg_pH10_output"])
print(val_data[idx]["deg_Mg_50C_output"])
print(val_data[idx]["deg_50C_output"])
train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_data,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)

valid_sampler = torch.utils.data.distributed.DistributedSampler(
          val_data,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)
TRAIN_BATCH_SIZE = 32

from torch.utils.data import DataLoader

training_dataloader = DataLoader(train_data,
                        num_workers=4,
                        batch_size=TRAIN_BATCH_SIZE,
                        sampler=train_sampler,
                        drop_last=True
                       )

val_dataloader = DataLoader(val_data,
                        num_workers=4,
                        batch_size=TRAIN_BATCH_SIZE,
                        sampler=valid_sampler,
                        drop_last=False
                       )
device = xm.xla_device()
!pip install efficientnet_pytorch

import efficientnet_pytorch

model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
import torch.nn as nn
import torch.nn.functional as F

# increasing few layers in our model
class EfficientNet_b0(nn.Module):
    def __init__(self):
        super(EfficientNet_b0, self).__init__()
        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
        
        self.image_dense_layer_1 = nn.Linear(1280 , 512)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout2d(0.5)
        self.image_dense_layer_2 = nn.Linear(512, 1)
        
        self.tabular_dense_layer_1 = nn.Linear(21, 16)
        self.tabular_dense_layer_2 = nn.Linear(16, 8)
        self.tabular_dense_layer_3 = nn.Linear(8, 4)
        self.tabular_dense_layer_4 = nn.Linear(4, 1)
        
        self.reactivity_layer = nn.Linear(2 , 1)
        self.deg_Mg_pH10_layer = nn.Linear(2 , 1)
        self.deg_pH10_layer = nn.Linear(2 , 1)
        self.deg_Mg_50C_layer = nn.Linear(2 , 1)
        self.deg_50C_layer = nn.Linear(2 , 1)
        
        
    def forward(self, image_inputs , tabular_data_inputs):
        x = self.model.extract_features(image_inputs)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        
        x = self.image_dense_layer_1(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.image_dense_layer_2(x)
        x = self.relu(x)
        
        tab = self.tabular_dense_layer_1(tabular_data_inputs)
        tab = self.relu(tab)
        tab = self.tabular_dense_layer_2(tab)
        tab = self.relu(tab)
        tab = self.tabular_dense_layer_3(tab)
        tab = self.relu(tab)
        tab = self.tabular_dense_layer_4(tab)
        tab = self.relu(tab)
        
        x = torch.cat((x, tab), dim=1)
        x = self.relu(x)

        return self.reactivity_layer(x) , self.deg_Mg_pH10_layer(x) , self.deg_pH10_layer(x), self.deg_Mg_50C_layer(x) , self.deg_50C_layer(x)
    
model = EfficientNet_b0()
model = model.to(device)
model
def loss_fn(predicted , actual):
    predicted_reactivity , predicted_deg_Mg_pH10 ,predicted_deg_pH10 , predicted_deg_Mg_50C , predicted_deg_50C = predicted 
    actual_reactivity , actual_deg_Mg_pH10 ,actual_deg_pH10 , actual_deg_Mg_50C , actual_deg_50C = actual
    
    reactivity_loss = torch.nn.MSELoss()(predicted_reactivity , actual_reactivity)
    deg_Mg_pH10_loss = torch.nn.MSELoss()(predicted_deg_Mg_pH10 , actual_deg_Mg_pH10)
    deg_pH10_loss = torch.nn.MSELoss()(predicted_deg_pH10 , actual_deg_pH10)
    deg_Mg_50C_loss = torch.nn.MSELoss()(predicted_deg_Mg_50C , actual_deg_Mg_50C)
    deg_50C_loss = torch.nn.MSELoss()(predicted_deg_50C , actual_deg_50C)
    
    return (reactivity_loss + deg_Mg_pH10_loss + deg_pH10_loss + deg_Mg_50C_loss + deg_50C_loss)/ 5
    
#for Stochastic Weight Averaging in PyTorch
from torchcontrib.optim import SWA

EPOCHS = 25
num_train_steps = int(len(train_data) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)

# printing the no of training steps for each epoch of our training dataloader  
xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')

params = list(model.image_dense_layer_1.parameters()) + \
         list(model.image_dense_layer_2.parameters()) + \
         list(model.tabular_dense_layer_1.parameters()) + \
         list(model.tabular_dense_layer_2.parameters()) + \
         list(model.tabular_dense_layer_3.parameters()) + \
         list(model.tabular_dense_layer_4.parameters()) + \
         list(model.reactivity_layer.parameters()) + \
         list(model.deg_Mg_pH10_layer.parameters()) + \
         list(model.deg_pH10_layer.parameters()) + \
         list(model.deg_Mg_50C_layer.parameters()) + \
         list(model.deg_50C_layer.parameters())

base_optimizer = torch.optim.Adam(params, lr=1e-4* xm.xrt_world_size())

optimizer = SWA(base_optimizer, swa_start=5, swa_freq=5, swa_lr=0.05)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# defining the training loop
def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    running_loss = 0.0
    model.train()
    
    for batch_index,dataset in enumerate(data_loader):
        image = dataset["image"]
        tabular_data = dataset["tabular_data"]
        reactivity_output = dataset["reactivity_output"]
        deg_Mg_pH10_output = dataset["deg_Mg_pH10_output"]
        deg_pH10_output = dataset["deg_pH10_output"]
        deg_Mg_50C_output = dataset["deg_Mg_50C_output"]
        deg_50C_output = dataset["deg_50C_output"]
        
        image = image.to(device, dtype=torch.float)
        tabular_data = tabular_data.to(device, dtype=torch.float)
        reactivity_output = reactivity_output.to(device, dtype=torch.float)
        deg_Mg_pH10_output = deg_Mg_pH10_output.to(device, dtype=torch.float)
        deg_pH10_output = deg_pH10_output.to(device, dtype=torch.float)
        deg_Mg_50C_output = deg_Mg_50C_output.to(device, dtype=torch.float)
        deg_50C_output = deg_50C_output.to(device, dtype=torch.float)
        
        optimizer.zero_grad()

        outputs = model(image, tabular_data)
        targets = (reactivity_output , deg_Mg_pH10_output , deg_pH10_output , deg_Mg_50C_output , deg_50C_output)
        loss = loss_fn(outputs , targets)

        loss.backward()
        xm.optimizer_step(optimizer)

        running_loss += loss.item()

    if scheduler is not None:
        scheduler.step()
            
    train_loss = running_loss / float(len(train_data))
    xm.master_print('training Loss: {:.4f}'.format(train_loss))
def eval_loop_fn(data_loader, model, device):
    running_loss = 0.0
    model.eval()
    
    for batch_index,dataset in enumerate(data_loader):
        image = dataset["image"]
        tabular_data = dataset["tabular_data"]
        reactivity_output = dataset["reactivity_output"]
        deg_Mg_pH10_output = dataset["deg_Mg_pH10_output"]
        deg_pH10_output = dataset["deg_pH10_output"]
        deg_Mg_50C_output = dataset["deg_Mg_50C_output"]
        deg_50C_output = dataset["deg_50C_output"]
        
        image = image.to(device, dtype=torch.float)
        tabular_data = tabular_data.to(device, dtype=torch.float)
        reactivity_output = reactivity_output.to(device, dtype=torch.float)
        deg_Mg_pH10_output = deg_Mg_pH10_output.to(device, dtype=torch.float)
        deg_pH10_output = deg_pH10_output.to(device, dtype=torch.float)
        deg_Mg_50C_output = deg_Mg_50C_output.to(device, dtype=torch.float)
        deg_50C_output = deg_50C_output.to(device, dtype=torch.float)
        

        outputs = model(image, tabular_data)
        targets = (reactivity_output , deg_Mg_pH10_output , deg_pH10_output , deg_Mg_50C_output , deg_50C_output)
        loss = loss_fn(outputs , targets)

        running_loss += loss.item()
    
    valid_loss = running_loss / float(len(val_data))
    xm.master_print('validation Loss: {:.4f}'.format(valid_loss))
def _run():
    for param in model.parameters():
        param.requires_grad = False
    
    for param in params:
        param.requires_grad = True
    
    for epoch in range(EPOCHS):
        xm.master_print(f"Epoch --> {epoch+1} / {EPOCHS}")
        xm.master_print(f"-------------------------------")
        para_loader = pl.ParallelLoader(training_dataloader, [device])
        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler)

        para_loader = pl.ParallelLoader(val_dataloader, [device])
        eval_loop_fn(para_loader.per_device_loader(device), model, device)
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = _run()
    optimizer.swap_swa_sgd()
    
# applying multiprocessing so that images get paralley trained in different cores of kaggle-tpu
FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=1, start_method='fork')
