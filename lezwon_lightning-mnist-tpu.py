# %%capture

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev


# %%capture

!pip uninstall -q typing --yes

!pip install -qU git+https://github.com/PyTorchLightning/pytorch-lightning.git

# !pip install -qU git+https://github.com/lezwon/pytorch-lightning.git@bugfix/2956_tpu_distrib_backend_fix
from pytorch_lightning import Trainer

from argparse import Namespace



import os



import torch

from torch.nn import functional as F

from torch.utils.data import DataLoader

from torchvision.datasets import MNIST

from torchvision import transforms



import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
class CoolSystem(pl.LightningModule):



    def __init__(self):

        super(CoolSystem, self).__init__()

        # not the best model...

        self.l1 = torch.nn.Linear(28 * 28, 10)



    def forward(self, x):

        # called with self(x)

        return torch.relu(self.l1(x.view(x.size(0), -1)))



    def training_step(self, batch, batch_idx):

        # REQUIRED

        x, y = batch

        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}



    def validation_step(self, batch, batch_idx):

        # OPTIONAL

        x, y = batch

        y_hat = self.forward(x)

        return {'val_loss': F.cross_entropy(y_hat, y)}



    def validation_end(self, outputs):

        # OPTIONAL

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

        

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=0.0004)



    def prepare_data(self):

        self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

        self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())



    def train_dataloader(self):

        loader = DataLoader(self.mnist_train, batch_size=32, num_workers=4)

        return loader



    def val_dataloader(self):

        loader = DataLoader(self.mnist_test, batch_size=32, num_workers=4)

        return loader





 
model = CoolSystem()

# logger = pl.loggers.TestTubeLogger(save_dir='./lightning_logs')

checkpoint_callback = pl.callbacks.ModelCheckpoint(

        filepath='checkpoints/tpu.ckpt',

        monitor='avg_val_loss',

        mode='min'

    )

trainer = Trainer(tpu_cores=8, precision=16, checkpoint_callback=False, max_epochs=5)

trainer.fit(model)  
!ls lightning_logs/default/
!ls checkpoints