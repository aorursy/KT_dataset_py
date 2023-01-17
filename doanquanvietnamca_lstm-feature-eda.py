import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

import seaborn as sns

import torch

import torch.nn as nn

from sklearn import preprocessing

from scipy.stats import norm, skew #for some statistics

from scipy import stats



import math

%matplotlib inline

plt.style.use('ggplot')
df_train = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train.csv")

df_train.head(5)
df_train.describe()
df_train.isnull().sum()
df_train.time_to_eruption.hist()
df_train.time_to_eruption.max()
sns.distplot(df_train.time_to_eruption , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train.time_to_eruption)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Time distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df_train.time_to_eruption, plot=plt)

plt.show()
df_first = pd.read_csv(f"../input/predict-volcanic-eruptions-ingv-oe/train/{df_train.segment_id[2]}.csv")

df_first.head()
df_first.describe()
df_first.sensor_2.hist()
%matplotlib inline



# calculate the correlation matrix

corr = df_first.corr()



# plot the heatmap

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
# MEAN = df_train.time_to_eruption.mean()

# NORM = np.linalg.norm(df_train.time_to_eruption.values)

# df_train.time_to_eruption= (df_train.time_to_eruption - MEAN)/ NORM
df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=42)
df_train[:60000]


class INDVDataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.segment = df.segment_id.values

        self.time = df.time_to_eruption.values

        self.limit_feature = np.ones((600,1))*10000

    def __len__(self):

        return self.segment.shape[0]

    def __getitem__(self, index):

        df = pd.read_csv(f"../input/predict-volcanic-eruptions-ingv-oe/train/{self.segment[index]}.csv")

        

        df = df.fillna(0)

        df = df[:60000].values.reshape(600, 1000)

#         x_scaled = df - self.limit_feature

#         x_scaled = x_scaled/(self.limit_feature)

        label = self.time[index]

        return df , label
BATCH_SIZE  = 10

NUM_WORKERS = 2

LR =1e-3

EPOCHS = 2
from torch.nn import functional as F

DEVICE= 'cpu'

class INGVNet(pl.LightningModule):

    def __init__(self):

        super().__init__()

        self.lstm1 = nn.LSTM(1000 , 100 , bidirectional=False, batch_first=True)

        self.linear1 = nn.Linear(200, 200)

        self.linear_aux_out = nn.Linear(200, 1)

        self.critrion = nn.MSELoss(reduction='mean')

        self.dropout = nn.Dropout(0.5)

        

    def forward(self, x):

        lstm1, _ = self.lstm1(x)

        avg_pool = torch.mean(lstm1, 1)

        max_pool, _ = torch.max(lstm1, 1)

        

        h_conc = torch.cat((max_pool, avg_pool), 1)

        h_conc_linear1  = self.linear1(h_conc)

        

        hidden = h_conc + h_conc_linear1

        hidden = self.dropout(hidden)

        aux_result = self.linear_aux_out(hidden)



        return aux_result





    def training_step(self, batch, batch_idx):

        # training_step defined the train loop. It is independent of forward

        x, y = [i.float().to(DEVICE) for i in batch]

        x_pred = self(x)

        loss = torch.sqrt(self.critrion(x_pred, y.reshape(-1, 1)))

        with torch.no_grad():

            logs = {

                'loss': loss,

                

            }

        return {'loss': loss, 'log': logs, "progress_bar": {"MAE": nn.L1Loss()(x_pred, y.reshape(-1, 1)) }}

    @torch.no_grad()

    def validation_step(self, batch, batch_idx):

        x, y = [i.float().to(DEVICE) for i in batch]

        x_pred = self(x)

        loss = self.critrion(x_pred, y.reshape(-1,1))

        logs = {

                'val_loss': loss

            }

        return logs

    def train_dataloader(self):

        train_dataset = INDVDataset(df_train)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, 

                                                   pin_memory=True, num_workers = NUM_WORKERS, shuffle=True)

        return train_loader

    def val_dataloader(self):

        valid_dataset = INDVDataset(df_valid)

        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, 

                                                   pin_memory=True, num_workers = NUM_WORKERS, shuffle=False)

        return valid_dataloader

        

    def configure_optimizers(self):

        self.optimizer = torch.optim.SGD(self.parameters(), lr=LR) #, betas= (0.9,0.999), weight_decay= 5e-7, amsgrad=True

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.6)

        return [self.optimizer], [self.scheduler]
net = INGVNet()
torch.backends.cudnn.benchmark =  True



trainer = pl.Trainer(max_epochs=EPOCHS ,gradient_clip_val=0, gpus=0)

trainer.fit(net)
df_test = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv")
class INDVTest(torch.utils.data.Dataset):

    def __init__(self, df):

        self.segment = df.segment_id.values

        self.limit_feature = np.ones((600,1))*10000

    def __len__(self):

        return self.segment.shape[0]

    def __getitem__(self, index):

        df = pd.read_csv(f"../input/predict-volcanic-eruptions-ingv-oe/test/{self.segment[index]}.csv")

        df = df.fillna(0)

        df = df[:60000].values.reshape(600, 1000)

#         x_scaled = df.values - self.limit_feature

#         x_scaled = x_scaled/(self.limit_feature)

        return df
from tqdm import tqdm

test_dataset = INDVTest(df_test)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, 

                                                   pin_memory=True, num_workers = NUM_WORKERS, shuffle=True)

arr_time = []

net.cuda()

for batch in tqdm(test_loader):

    arr_time= [*arr_time, *  (net(batch.float().to(DEVICE)).squeeze().detach().cpu().numpy()*NORM+MEAN)]

df_test.time_to_eruption = arr_time
df_test.to_csv("submission.csv", index=False)