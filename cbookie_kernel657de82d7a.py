import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd

import math

from matplotlib import pyplot as plt

%matplotlib inline
DATA_CSV = '/kaggle/input/avocado-prices/avocado.csv'
train_df = pd.read_csv(DATA_CSV)
train_df.head()
class AvocadoDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        inputs = torch.Tensor([row['4046'], row['4225'], row['4770'], row['Total Bags'], row['Small Bags'], row['Large Bags'], row['XLarge Bags'], row['year']]).to(torch.float)
        target = torch.tensor(row['AveragePrice'], dtype=torch.float32)
        return inputs, target

using_GPU = torch.cuda.is_available()
if using_GPU:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def to_device(data):
    if isinstance(data, (list,tuple)):
        return [to_device(x) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, ds, batch_size, Training=False):        
        self.dl = DataLoader(ds, batch_size, shuffle=Training, num_workers=2, pin_memory=using_GPU)
    
    def __iter__(self):
        return (to_device(data) if using_GPU else data for data in self.dl)
#         for data in self.dl:
#             if using_GPU:
#                 yield to_device(data)
#             else:
#                 yield data

    def __len__(self):
        return len(self.dl)
dataset = AvocadoDataset(DATA_CSV)
len(dataset)
val_pct = 0.1
val_size = int(val_pct * len(dataset))
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)
batch_size = 32
train_dl = DeviceDataLoader(train_ds, batch_size, True)
val_dl = DeviceDataLoader(val_ds, batch_size*2)
from tqdm.notebook import tqdm
class AvocadoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
#             nn.BatchNorm1d(),
            nn.Linear(8, 16),
            nn.LeakyReLU(),
            nn.Dropout(),

            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Dropout(),

            nn.Linear(32, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 16),
            nn.LeakyReLU(),

            nn.Linear(16, 8),
            nn.LeakyReLU(),

            nn.Linear(8, 1)
        )
        
    def forward(self, xb):
        return self.network(xb)
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        self.eval()
        outputs = list()
        for batch in val_loader:
            inputs, targets = batch
            out = self(inputs)
            loss = F.mse_loss(out, targets.to(torch.float))
            outputs.append({'val_loss': loss.detach()})
            
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}



    def fit(self, epochs, lr, train_loader, val_loader, opt_func=torch.optim.SGD):
        torch.cuda.empty_cache()
        history = []
        optimizer = opt_func(self.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase 
            self.train()
            train_losses = []
            for batch in tqdm(train_loader):
#             for batch in train_loader:
                inputs, targets = batch
                out = self(inputs)
                loss = F.mse_loss(out, targets.to(torch.float)).to(torch.float)
                
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            result = self.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result['train_loss'], result['val_loss']))
            history.append(result)
        return history
model = AvocadoModel()
model = model.to(device, non_blocking=True)
model.evaluate(val_dl)
num_epochs = 50
opt_func = torch.optim.Adam
lr = 1e-2
history = model.fit(num_epochs, lr, train_dl, val_dl, opt_func)
plt.plot(range(len(history)), [math.log(slice['train_loss']) for slice in history])
plt.show()
plt.plot(range(len(history)), [math.log(slice['val_loss']) for slice in history])
plt.show()
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project='avocado-prices')