# !pip install colabcode
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
# from colabcode import ColabCode
# ColabCode()
import pandas as pd

df = pd.read_csv("../input/landmark-recognition-2020/train.csv")
from pathlib import Path

train_files = Path('../input/landmark-recognition-2020/train').glob('**/*.jpg')
test_files = Path('../input/landmark-recognition-2020/test').glob('**/*.jpg')
# test_files = glob.glob('../input/landmark-recognition-2020/test/*/*/*/*.jpg')
# create folds
from sklearn import model_selection

df["kfold"] = -1    
df = df.sample(frac=1).reset_index(drop=True)
y = df.landmark_id.values
kf = model_selection.StratifiedKFold(n_splits=5)

for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f
!pip uninstall typing -y
!pip install -qU git+https://github.com/PyTorchLightning/pytorch-lightning.git
!pip install torchtoolbox
from PIL import Image
import cv2
import albumentations
import torch
import numpy as np
import io
from torch.utils.data import Dataset
from torchtoolbox.transform import Cutout

# Making the dataset class for training and testing Flower images

class Landmark_detection_Dataset(Dataset):
    def __init__(self, id , classes , image , img_height , img_width, mean , std , is_valid):
        self.id = id
        self.classes = classes
        self.image = image
        self.is_valid = is_valid
        if self.is_valid == 1: # transforms for validation images
            self.aug = albumentations.Compose([
               albumentations.Resize(img_height , img_width, always_apply = True) ,
               albumentations.Normalize(mean , std , always_apply = True) 
            ])
        else:                  # transfoms for training images 
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height , img_width, always_apply = True) ,
                albumentations.Normalize(mean , std , always_apply = True),
                albumentations.ShiftScaleRotate(shift_limit = 0.0625,
                                                scale_limit = 0.1 ,
                                                rotate_limit = 5,
                                                p = 0.9)
            ]) 
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, index):
        id = self.id[index]
        
        # converting jpg format of images to numpy array
        img = np.array(Image.open('../input/landmark-recognition-2020/train/'+
                                  self.image[index][0]+'/'+
                                  self.image[index][1]+'/'+
                                  self.image[index][2]+'/'+
                                  self.image[index]+'.jpg')) 
        
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        img = self.aug(image = img)['image']
        img = np.transpose(img , (2,0,1)).astype(np.float32) # 2,0,1 because pytorch excepts image channel first then dimension of image
       
        return torch.tensor(img, dtype = torch.float),torch.tensor(self.classes[index], dtype = torch.long)
    
fold = 0
df_train = df[df.kfold != fold].reset_index(drop=True)
df_valid = df[df.kfold == fold].reset_index(drop=True)

# prepare transforms standard to MNIST
train_data = Landmark_detection_Dataset(id = [i for i in range(len(df_train))], 
                                         classes = df_train['landmark_id'], 
                                         image = df_train['id'], 
                                         img_height = 224 , img_width = 224, 
                                         mean = (0.485, 0.456, 0.406),
                                         std = (0.229, 0.224, 0.225) , is_valid = 0)

val_data = Landmark_detection_Dataset(id = [i for i in range(len(df_valid))], 
                                       classes = df_valid['landmark_id'], 
                                       image = df_valid['id'], 
                                       img_height = 224 , img_width = 224, 
                                       mean = (0.485, 0.456, 0.406),
                                       std = (0.229, 0.224, 0.225) , is_valid = 1)
import matplotlib.pyplot as plt
%matplotlib inline

idx = 1000 # taking index for 10000th image out of 51000 images
img = val_data[idx][0]

print(val_data[idx][1]) # val_dataset label is one Hot encoded

npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1,2,0)))
from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, batch_size=16, num_workers=4)
val_loader = DataLoader(val_data, batch_size=16, num_workers=4)
!pip install pretrainedmodels
import pretrainedmodels
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms

no_of_outputs_classes_for_our_dataset = len(df['landmark_id'].unique())

class ResNet34(pl.LightningModule):
    
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model =  pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')

        self.last_dense_layer = torch.nn.Linear(self.model.last_linear.in_features, no_of_outputs_classes_for_our_dataset)
        
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size ,_,_,_ = x.shape     #taking out batch_size from input image
        x = self.model.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)     # then reshaping the batch_size
        x = self.last_dense_layer(x)
        return x
    
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        predictions = self.forward(x)
        loss = self.loss(predictions, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        predictions = self.forward(x)
        loss = self.loss(predictions, y)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.last_dense_layer.parameters(), lr=1e-3)
        return optimizer

from pytorch_lightning import Trainer, seed_everything
seed_everything(0)

model = ResNet34()                 

trainer = Trainer(tpu_cores=8, progress_bar_refresh_rate=20, max_epochs=10)
trainer.fit(model, train_loader, val_loader)
!pip install torchcontrib
#for Stochastic Weight Averaging in PyTorch
from torchcontrib.optim import SWA

base_optimizer = torch.optim.Adam(model.last_dense_layer.parameters(), lr=1e-4)

optimizer = SWA(base_optimizer, swa_start=5, swa_freq=5, swa_lr=0.05)

loss_fn = torch.nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
