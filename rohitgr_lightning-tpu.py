!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
!pip uninstall -q typing --yes
# !pip install -U git+https://github.com/lezwon/pytorch-lightning.git@2016_test
# !pip install -qU git+https://github.com/rohitgr7/pytorch-lightning@fix_tpu_id
# !pip install -U git+https://github.com/lezwon/pytorch-lightning.git@bugfix/2016_slow_tpu_train
!pip install -qU git+https://github.com/PyTorchLightning/pytorch-lightning.git
!rm -rf '/kaggle/working/lightning_logs'
import os

import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import KFold
import torch_xla.core.xla_model as xm


import torch_xla
# Set a seed for numpy for a consistent Kfold split
np.random.seed(123)
# Download the dataset in advance
MNIST(os.getcwd(), train=True, download=True)
MNIST(os.getcwd(), train=False, download=True)
class MNISTModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super(MNISTModel, self).__init__()
#         self.hparams = hparams
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=6.918309709189366e-07)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        # print(self.device, next(self.parameters()).device)
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': {'tpu': torch_xla._XLAC._xla_get_default_device()}}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    # def test_step(self, batch, batch_nb):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self(x)
    #     return {'test_loss': F.cross_entropy(y_hat, y)}

    # def test_epoch_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     logs = {'test_loss': avg_loss}
    #     return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    
    def prepare_data(self):
        dataset = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
        self.mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor())
        
        kf = KFold(n_splits=8)
        splits = list(kf.split(dataset))

        train_indices, val_indices = splits[1]
        
        self.mnist_train = torch.utils.data.Subset(dataset, train_indices)
        self.mnist_val = torch.utils.data.Subset(dataset, val_indices)
                
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = cls()
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def train_dataloader(self):
        # loader = DataLoader(self.mnist_train, batch_size=32, num_workers=4)
        # return loader
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
        loader = DataLoader(mnist_train, batch_size=32, num_workers=4)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.mnist_val, batch_size=32, num_workers=4)
        return loader

    # def test_dataloader(self):
        # loader = DataLoader(self.mnist_test, batch_size=32, num_workers=4)
        # return loader
# mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
# loader = DataLoader(mnist_train, batch_size=32, num_workers=4)

# for a in loader:
#     print(a[0].shape)
# #     break
hparams = {'lr': 6.918309709189366e-07, 'fold': 1}
# model = MNISTModel(hparams)
# trainer = pl.Trainer(tpu_cores=8, max_epochs=1)    
# Define a function to initialize and train a model
def train(tpu_id):
    model = MNISTModel(hparams)
    trainer = pl.Trainer(tpu_cores=tpu_id, max_epochs=2, checkpoint_callback=True, weights_summary=None)    
    trainer.fit(model)
    print('Training Done')
#     trainer.test(model, ckpt_path=None)
# Specifying tpu core id
# train([1])
# Training on 1 core
# train(1)
# Specifying tpu core id
train(8)
# #use joblib to run the train function in parallel on different folds
# import joblib as jl
# parallel = jl.Parallel(n_jobs=8, backend="threading", batch_size=1)
# parallel(jl.delayed(train)(i+1) for i in range(8))
# weights are saved to checkpoints
# !ls -lh checkpoints/ 
# xm.xla_device(n=8, devkind='TPU')
# dev = xm.xla_device()
# t1 = torch.ones(3, 3, device = dev)
# print(t1)