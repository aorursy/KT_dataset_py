from google.colab import drive
drive.mount('/content/drive')
#!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
#!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
!nvidia-smi 
!pip install rasterio
!pip install pytorch-lightning-bolts
!pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@master --upgrade
import numpy as np
import pandas as pd
import rasterio

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.metrics import MeanAbsoluteError
#import albumentations as A

#from sklearn.preprocessing import MinMaxScaler    
from sklearn.metrics import r2_score

import glob
import os

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns
csv_path = '/content/drive/My Drive/hdi_with_geometry.csv'
root_dir = '/content/drive/My Drive/Images/'
model =  torchvision.models.resnet18(pretrained=False, progress=True)
model.conv1 = nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)
model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=1000, bias=True), nn.Sigmoid())
class MyDataset(Dataset):
    """
    Generate normalized, rescaled and transformed datasets
    """

    def __init__(self, dataset, transform=None):
        
        super().__init__()
        self.df = dataset
        self.transform = transform

    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
       
        if torch.is_tensor(idx):
              idx = idx.tolist()

        # generate image sample
        image_path = self.df['image_path'].iloc[idx] 
        image_sample = rasterio.open(str(image_path), "r")
        bands = [i for i in range(1, image_sample.count+1)]
        image_sample = image_sample.read(bands)
        image_sample = image_sample.astype('float32')

        # generate hdi sample

        hdi_sample = self.df['HDI'].iloc[idx]

        # Normalize the image sample and rescale it between 0 and 1
        for ch in range(image_sample.shape[0]):
            channel_mean = np.nanmean(image_sample[ch])
            channel_stdev = np.nanstd(image_sample[ch])
            image_sample[ch] = (image_sample[ch] - channel_mean)

            if channel_stdev != 0:

                # standardize
                image_sample[ch] = image_sample[ch] / channel_stdev
                
                # normalize
                image_sample[ch] = (image_sample[ch] - np.nanmin(image_sample[ch])) / (np.nanmax(image_sample[ch]) - np.nanmin(image_sample[ch]))
        
        # convet nan to 0
        image_sample[np.isnan(image_sample)] = 0

        if self.transform:
            image_sample = self.transform(image_sample)

        else:
            return [image_sample.permute(1, 0, 2),  hdi_sample.astype('float32')]
    
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor,  ModelCheckpoint

# default used by the Trainer
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    strict=False,
    verbose=True,
    mode='min')


# DEFAULTS used by the Trainer
checkpoint_callback = ModelCheckpoint(
    filepath='/content/drive/My Drive/ckpt/model.ckpt',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

lr_monitor = LearningRateMonitor(logging_interval='step')

# custome collate to pad images for each batch
def my_collate(batch):

    max_wh = 0

    for item in batch:
        image = item[0]
        w = image.shape[1]
        h = image.shape[2]
        max_i = np.max([w, h])
        if max_i > max_wh:
            max_wh = max_i
    
    #print(max_wh)

    data = []

    for item in batch:
        image = item[0]
        rows = image.shape[1]
        cols = image.shape[2]
        rows_diff = max_wh - rows
        cols_diff = max_wh - cols
        cols_half = int(cols_diff / 2)
        rows_half = int(rows_diff / 2)
        padding = (cols_half, cols_diff-cols_half, rows_half, rows_diff-rows_half)
        image_pad = F.pad(image, padding, 'constant', 0)
        data.append(image_pad)

    target = [item[1] for item in batch]
    return [data, target]

class Model(pl.LightningModule):

    def __init__(self, model, batch_size=1, learning_rate=.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.model = model
        self.batch_size = batch_size
        self.ser_y = pd.Series(dtype='float32', name='y')
        self.ser_y_hat = pd.Series(dtype='float32', name='y_pred')

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, 1)
        return x

    def prepare_data(self):

        df = pd.read_csv(csv_path)
        df['image_path'] = root_dir + df['unique code'].astype(str) + '.tif' 
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        # split the dataset
        train, validate, test = np.split(df, [int(.9*len(df)), int(.95*len(df))]) 

        # transforms
        train_transform = transforms.Compose([
                                transforms.ToTensor()
                                ])

        validate_transform = transforms.Compose([
                                transforms.ToTensor()
                                ])
        # create datasets for training, validation and test
        self.train_dataset = MyDataset(dataset=train, transform=train_transform)
        self.validate_dataset = MyDataset(dataset=validate, transform=validate_transform)
        self.test_dataset = MyDataset(dataset=test, transform=validate_transform) 

        return test
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validate_dataset, self.batch_size, num_workers=4,collate_fn=my_collate, pin_memory=True, drop_last=True) 

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, num_workers=4, collate_fn=my_collate, pin_memory=True, drop_last=True) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.stack(x)
        y = torch.cuda.FloatTensor(y)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss



    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = torch.stack(x)
        y = torch.cuda.FloatTensor(y)
        y_hat = self(x)
        #print(y, y_hat)
        loss = F.mse_loss(y_hat, y)
        r2 = r2_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
        self.log('val_loss', loss)
        self.log('val_R-square', r2)
        #return {"loss": loss, 'R-square': r2_score}


    def test_step(self, batch, batch_idx):
        x, y = batch

        y_series = pd.Series(y)
        
        x = torch.stack(x)
        y = torch.cuda.FloatTensor(y)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        r2 = r2_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
        
        self.log('test_loss', loss)
        self.log('test_R-square', r2)

        y_hat = y_hat.cpu().detach().numpy()
        y_hat_series = pd.Series(y_hat)

        self.ser_y = self.ser_y.append(y_series, ignore_index=True)
        self.ser_y_hat = self.ser_y_hat.append(y_hat_series, ignore_index=True)
    

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def optimizer_step(self, current_epoch, batch_idx, optimizer, 
      optimizer_idx, second_order_closure=None, 
       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step()

# init model
model_one = Model(model, batch_size=16)
#from pytorch_lightning.core.memory import ModelSummary
#ModelSummary(model_one, mode='full')
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger('/content/drive/My Drive/tb_logs', name='my_model')

#train
root_path = '/content/drive/My Drive/'
#seed
resume_ckpt_path =  '/content/drive/My Drive/ckpt/model_t3.ckpt'
#resume_from_checkpoint=resume_ckpt_path,
pl.seed_everything(1)

trainer = pl.Trainer(gpus=1,resume_from_checkpoint=resume_ckpt_path, logger=logger, checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=50, accumulate_grad_batches=2, fast_dev_run=False,\
                    default_root_dir=root_path, auto_lr_find=True,\
                    profiler=True, max_epochs=1000, callbacks=[lr_monitor, early_stop, PrintTableMetricsCallback()])



trainer.fit(model_one)
#!nvidia-smi 
# Start tensorboard.
%reload_ext tensorboard
%tensorboard --logdir='/content/drive/My Drive/tb_logs'
# test
trainer.test(ckpt_path='/content/drive/My Drive/ckpt/model.ckpt-v0.ckpt', model=model_one, verbose=True)
# Analysis
df_pred = pd.concat([model_one.ser_y, model_one.ser_y_hat], axis=1)
df_pred = df_pred.set_axis(['y', 'y_pred'], axis=1, inplace=False)
df_pred['diff'] = (df_pred.y - df_pred.y_pred).abs()
df_pred
df = pd.read_csv(csv_path)
df['image_path'] = root_dir + df['unique code'].astype(str) + '.tif' 
df = df.sample(frac=1, random_state=1).reset_index(drop=True)
# split the dataset
train, validate, test = np.split(df, [int(.9*len(df)), int(.95*len(df))]) 
test.head(10)
test.reset_index(inplace=True, drop=True)
df_final = pd.concat([test, df_pred], axis =1)
df_final = df_final[df_final['y_pred'].notna()]
df_final.sort_values(by=['diff'], ascending=False, inplace=True)
df_final['num_pixel'] = df['SHAPE_Area'] / 900
df_final.reset_index(inplace=True, drop=True)
df_final.head()
df_final.tail()
df_final.to_csv('/content/drive/My Drive/post_training_analysis.csv')
df_final = pd.read_csv('/content/drive/My Drive/post_training_analysis.csv')
df_final.describe()
import matplotlib.pyplot as plt
from rasterio.plot import show

plt.figure(figsize=(10,8))
# the file has been downloaded from drive to show here
image_h = rasterio.open(df_final['image_path'].iloc[4])
show(image_h, adjust='linear')
plt.show()
plt.figure(figsize=(10,8))
# the file has been downloaded from drive to show here
image = rasterio.open(df_final['image_path'].iloc[-2])
show(image, adjust='linear')
plt.show()
sns.distplot(df_final['y'])
sns.distplot(df_final['y_pred']);
sns.distplot(df_final['diff'])
df_corr = df_final.corr()# irrelevant fields
fields = ['unique code', 'HDI']# drop rows
df_corr.drop(fields, inplace=True)# drop cols
df_corr.drop(fields, axis=1, inplace=True)
df_corr