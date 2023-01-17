# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import torch

import torch.nn as nn

import torch.nn.functional as F

from tqdm import tqdm_notebook

from torch.utils.data import DataLoader, Dataset

import ignite

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from ignite.metrics import Accuracy, Loss

from ignite.contrib.handlers import TensorboardLogger, ProgressBar

from torch.optim.lr_scheduler import ExponentialLR

from ignite.engine.engine import Engine, State, Events

from ignite.utils import convert_tensor

import gc
%%writefile preprocess.py

import numpy as np

import pandas as pd

import gc

from tqdm import tqdm_notebook

def preprocess():

    train = pd.read_csv('/kaggle/input/LANL-Earthquake-Prediction/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

    etq_meta = [

    {"start":0,         "end":5656574},

    {"start":5656574,   "end":50085878},

    {"start":50085878,  "end":104677356},

    {"start":104677356, "end":138772453},

    {"start":138772453, "end":187641820},

    {"start":187641820, "end":218652630},

    {"start":218652630, "end":245829585},

    {"start":245829585, "end":307838917},

    {"start":307838917, "end":338276287},

    {"start":338276287, "end":375377848},

    {"start":375377848, "end":419368880},

    {"start":419368880, "end":461811623},

    {"start":461811623, "end":495800225},

    {"start":495800225, "end":528777115},

    {"start":528777115, "end":585568144},

    {"start":585568144, "end":621985673},

    {"start":621985673, "end":629145480},

    ]



    df = []

    for i in [2, 7, 0, 4, 11, 13, 9, 1, 14, 10]:

        df.append(train[etq_meta[i]['start']:etq_meta[i]['start']+150000*((etq_meta[i]['end'] - etq_meta[i]['start'])//150000)])

    

    train = pd.concat(df)



    num_seg = len(train)//150000

    train_X = []

    train_y = []

    for i in tqdm_notebook(range(num_seg)):

    #     train_X.append(fft_process(train['acoustic_data'].iloc[150000 * i:150000 * i + 150000]))

        if 100000 * i + 150000 < len(train):

            train_X.append(train['acoustic_data'].iloc[150000 * i:150000 * i + 150000])

            train_y.append(train['time_to_failure'].iloc[150000 * i + 149999])

    del train

    gc.collect()

    train_X = np.array(train_X,dtype = np.float32)

    train_y = np.array(train_y,dtype = np.float32)



    X_mean = train_X.mean(0)

    X_std = train_X.std(0)

    train_X -= X_mean

    train_X /= X_std

    y_mean = train_y.mean()

    y_std = train_y.std()

    train_y -= y_mean

    train_y /= y_std



    train_X = np.expand_dims(train_X,-1)

    np.save('train_x.npy',train_X)

    np.save('train_y.npy',train_y)

    gc.collect()
from preprocess import preprocess

preprocess()

train_X = np.load('train_x.npy')

train_y = np.load('train_y.npy')
class Attention(nn.Module):

    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):

        super(Attention, self).__init__(**kwargs)

        

        self.supports_masking = True



        self.bias = bias

        self.feature_dim = feature_dim

        self.step_dim = step_dim

        self.features_dim = 0

        

        weight = torch.zeros(feature_dim, 1)

        nn.init.xavier_uniform_(weight)

        self.weight = nn.Parameter(weight)

        

        if bias:

            self.b = nn.Parameter(torch.zeros(step_dim))

        

    def forward(self, x, mask=None):

        feature_dim = self.feature_dim

        step_dim = self.step_dim



        eij = torch.mm(

            x.contiguous().view(-1, feature_dim), 

            self.weight

        ).view(-1, step_dim)

        

        if self.bias:

            eij = eij + self.b

            

        eij = torch.tanh(eij)

        a = torch.exp(eij)

        

        if mask is not None:

            a = a * mask



        a = a / torch.sum(a, 1, keepdim=True) + 1e-10



        weighted_input = x * torch.unsqueeze(a, -1)

        return torch.sum(weighted_input, 1)
class Wave_Block(nn.Module):

    

    def __init__(self,in_channels,out_channels,dilation_rates):

        super(Wave_Block,self).__init__()

        self.num_rates = dilation_rates

        self.convs = nn.ModuleList()

        self.filter_convs = nn.ModuleList()

        self.gate_convs = nn.ModuleList()

        

        self.convs.append(nn.Conv1d(in_channels,out_channels,kernel_size=1))

        dilation_rates = [2**i for i in range(dilation_rates)]

        for dilation_rate in dilation_rates:

            self.filter_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=dilation_rate,dilation=dilation_rate))

            self.gate_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=dilation_rate,dilation=dilation_rate))

            self.convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=1))

            

    def forward(self,x):

        x = self.convs[0](x)

        res = x

        for i in range(self.num_rates):

            x = F.tanh(self.filter_convs[i](x))*F.sigmoid(self.gate_convs[i](x))

            x = self.convs[i+1](x)

            x += res

        return x
class Wave_LSTM(nn.Module):

    def __init__(self):

        super().__init__()

        in_channels = 150000



        self.LSTM = nn.GRU(input_size=in_channels//10**3,hidden_size=in_channels//10**3,num_layers=64,batch_first=True,bidirectional=True)

        self.conv1 = nn.Linear(300,128)

        self.conv2 = nn.Linear(128,1)

        self.attention = Attention(300,64)

        self.avgpool1d = nn.AvgPool1d(10)

        self.wave_block1 = Wave_Block(1,16,8)

        self.wave_block2 = Wave_Block(16,32,5)

        self.wave_block3 = Wave_Block(32,64,3)

            

    def forward(self,x):

        x = self.wave_block1(x)

        #shrinking

        x = self.avgpool1d(x)

        x = self.wave_block2(x)

        #shrinking

        x = self.avgpool1d(x)

        x = self.wave_block3(x)

        #shrinking

        x = self.avgpool1d(x)

        x,_ = self.LSTM(x)

        x = self.attention(x)

        x = F.dropout(x,0.2)

        x = self.conv1(x)

        x = self.conv2(x)

        return x
!pip install torchsummary

from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Wave_LSTM().to(device)

summary(model.to(device), input_size=(1, 150000))
class Dataset(Dataset):

    def __init__(self,features,target):

        super().__init__()

        self.features = features

        self.target = target

        

    def __len__(self):

        return len(self.target)

    

    def __getitem__(self,idx):

        feat = self.features[idx]

        trg = self.target[idx]

        

        return feat,trg
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.10, random_state=42)
def get_data_loaders(batch_size=32):

    train_dataset = Dataset(X_train,y_train)

    val_dataset = Dataset(X_test,y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True)

    return train_loader,val_loader

train_loader , val_loader = get_data_loaders()

gc.collect()
def _prepare_batch(batch, device=None, non_blocking=False):

    """Prepare batch for training: pass to a device with options.



    """

    x, y = batch

    return (convert_tensor(x, device=device, non_blocking=non_blocking),

            convert_tensor(y, device=device, non_blocking=non_blocking))

def create_supervised_trainer1(model, optimizer, loss_fn, metrics={}, device=None):



    def _update(engine, batch):

        model.train()

        optimizer.zero_grad()

        x, y = _prepare_batch(batch, device=device)

        y_pred = model(x.permute(0,2,1))

        loss = loss_fn(y_pred, y)

        loss.backward()

        optimizer.step()

        return loss.item(), y_pred, y



    def _metrics_transform(output):

        return output[1], output[2]



    engine = Engine(_update)



    for name, metric in metrics.items():

        metric._output_transform = _metrics_transform

        metric.attach(engine, name)



    return engine



def create_supervised_evaluator1(model, metrics=None,

                                device=None, non_blocking=False,

                                prepare_batch=_prepare_batch,

                                output_transform=lambda x, y, y_pred: (y_pred, y,)):

    """

    Factory function for creating an evaluator for supervised models.



    Args:

        model (`torch.nn.Module`): the model to train.

        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.

        device (str, optional): device type specification (default: None).

            Applies to both model and batches.

        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously

            with respect to the host. For other cases, this argument has no effect.

        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs

            tuple of tensors `(batch_x, batch_y)`.

        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value

            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits

            output expected by metrics. If you change it you should use `output_transform` in metrics.



    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is

        a tuple of `(batch_pred, batch_y)` by default.



    Returns:

        Engine: an evaluator engine with supervised inference function.

    """

    metrics = metrics or {}



    if device:

        model



    def _inference(engine, batch):

        model.eval()

        with torch.no_grad():

            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

            y_pred = model(x.permute(0,2,1))

            return output_transform(x, y, y_pred)



    engine = Engine(_inference)



    for name, metric in metrics.items():

        metric.attach(engine, name)



    return engine
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

metrics = {

    'loss': Loss(criterion),

}

model = Wave_LSTM()

trainer = create_supervised_trainer1(model.to(device), optimizer, criterion, device=device)

val_evaluator = create_supervised_evaluator1(model.to(device), metrics=metrics, device=device)

@trainer.on(Events.EPOCH_COMPLETED)

def compute_and_display_val_metrics(engine):

    epoch = engine.state.epoch

    metrics = val_evaluator.run(val_loader).metrics

    print("Validation Results - Epoch: {}  Average Loss: {:.4f}"

          .format(engine.state.epoch, 

                      metrics['loss']))

pbar = ProgressBar(bar_format='')

pbar.attach(trainer, output_transform=lambda x: {'loss': x})



lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

@trainer.on(Events.EPOCH_COMPLETED)

def update_lr_scheduler(engine):

    lr_scheduler.step()

    lr = float(optimizer.param_groups[0]['lr'])

    print("Learning rate: {}".format(lr))

gc.collect()
trainer.run(train_loader, max_epochs=10)