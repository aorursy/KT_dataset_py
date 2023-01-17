import torch

from pathlib import Path

import gzip, pickle

from torch import nn

from torch import tensor

from torch.utils.data import DataLoader

from torch import optim

import os

import copy

from math import ceil

import math

from torch.nn import init

from functools import partial

from torch.nn import functional as F

Path.ls = lambda x: list(x.iterdir())

import pandas as pd

import numpy as np



path = Path('/kaggle/input/digit-recognizer/')



def get_data(path,fn='train.csv'):

    df = pd.read_csv(path/fn)

    if 'label' not in df.columns:

        vals = np.ones_like(df.iloc[:,0].values)*-1

        df.insert(0,'label',vals)

    X = df.iloc[:,1:].values

    y = df.iloc[:,0].values

    return X,y





class Dataset:

    def __init__(self,X,y):

        self.X, self.y = X,y

        

    def __len__(self):

        return len(self.X)

    

    def __getitem__(self,idx):

        return torch.tensor(self.X[idx],dtype=torch.float),torch.tensor(self.y[idx],dtype=torch.long)



def get_dls(train_ds,test_ds,bs=64):

    return (DataLoader(train_ds,batch_size = bs, shuffle=True, drop_last=True),

            DataLoader(test_ds, batch_size = bs*2, shuffle=False))



def init_cnn(m, uniform=False):

    f = init.kaiming_uniform_ if uniform else init.kaiming_normal_

    for l in m:

        if isinstance(l, nn.Sequential):

            f(l[0].weight, a=0.1)

            if l[0].bias is not None:

                l[0].bias.data.zero_()



class Lambda(nn.Module):

    """

    Custom nn.Module to pass functions like flatten and resize for mnist

    """

    def __init__(self,func):

        super().__init__()

        self.func = func

    

    def forward(self,x):

        return self.func(x)



def flatten(x):

    return x.view(x.shape[0],-1)



def mnist_resize(x):

    return x.view(-1,1,28,28)



def get_model(layers,pct_sparsity=0.,use_gpu=True):

    """

    Creates a Pytorch nn.Sequential Model with the specified layers and a SGD optimizer.

    

    Args: 

        use_gpu: (boolean) If True loads the model on the GPU. If False, uses CPU

        

    Returns:

        Model and optimizer

    """

    # pct_units = round(1 - pct_sparsity,2)

    model = layers;

    

    if use_gpu:

        return model.cuda()

    

    return model

    



def get_optimizer(model,lr=0.1):

    """

    Creates SGD optimizer with the model parameters and learning rate

    

    Args:

        model : nn.Module model 

        lr (float): learning rate for the optimizer

        

    Returns: 

        returns an optimizer

    """

    

    return optim.SGD(model.parameters(), lr=lr, momentum=0.9)



def accuracy(preds, y):

    return (torch.argmax(preds, dim=1) == y).float().mean()



def average_metrics(dataloader,model,metrics=[],use_gpu=True):

    """

    Calculates the weighted average metrics, weighted by the batch size.

    This ensures the correctness of the average metrics calculated even if the batch size changes

    

    Args: 

        dataloader (DataLoader) : DataLoader for the required dataset. Preferably the validation set.

        metrics (list) : A list of metrics. Each metric passed in the list should be a function.

        use_gpu: (boolean) If True loads the data batch on the GPU. If False, keeps the data batch on CPU



    Returns:

        List of average metrics in the order passed in the metrics list.

        

    """

    with torch.no_grad():

        count = 0

        tot_metrics = [0. for _ in metrics]

        for xb,yb in dataloader:

            if use_gpu:

                xb, yb = xb.cuda(), yb.cuda()

                

            bs=len(xb)

            for idx, metric in enumerate(metrics):

                tot_metrics[idx] += metric(model(xb),yb) * bs  # metric * batch_size for weighted average



            count += bs



        avg_metrics = list()

        for metric in tot_metrics:

            avg_metrics.append(metric/count)



        return avg_metrics



def fit_one_cycle(epochs,sched_func,use_gpu=True):

    """

    The fit function contains the training loop. Prints the validation accuracy and validation loss

    

    Args: 

        epochs: Number of epochs to fit

        use_gpu: (boolean) If True loads the data batch on the GPU. If False, keeps the data batch on CPU

    

    """

    n_epochs = 0;

    

    for epoch in range(epochs):       # for each epoch

        n_epochs = epoch

        iters = len(train_dl)

        model.train();

        for xb,yb in train_dl:        # for each batch in the train_dl

            if use_gpu:

                xb, yb = xb.cuda(), yb.cuda()

            preds = model(xb)  # predictions are calculated in the forward pass

            loss = loss_func(preds,yb)  # loss is calculated

            loss.backward()           # the gradients are accumulated

            sched_params(opt, n_epochs, epochs,sched_func)

            opt.step()                # weights are updated by 'lr * gradients'

            opt.zero_grad()           # gradients are set to zeros

            n_epochs += 1./iters

        print(f"Epoch {epoch} completed")

#         model.eval()

#         with torch.no_grad():

# #           # print(f'epoch : {epoch} train acc: {average_metrics(train_dl,model,[accuracy,loss_func],use_gpu)}')

#           print(f'epoch : {epoch} validation acc: {average_metrics(valid_dl,model,[accuracy,loss_func],use_gpu)}')



def sched_params(opt, n_epochs, epoch,sched_func):

    for pg in opt.param_groups:

        pg['lr'] = sched_func(n_epochs/epoch)



def annealer(f):

    def _inner(start, end): return partial(f, start, end)

    return _inner



@annealer

def sched_cos(start, end, pos): return start + (1+math.cos(math.pi*(1-pos))) * (end-start)/2



def combine_scheds(pcts, scheds):

    assert sum(pcts) == 1

    pcts = tensor([0]+ list(pcts))

    assert torch.all(pcts>=0)

    pcts = torch.cumsum(pcts,0)

    def _inner(pos):

        idx = (pos>=pcts).nonzero().max()

        actual_pos = (pos-pcts[idx])/(pcts[idx+1]-pcts[idx])

        return scheds[idx](actual_pos)

    return _inner



def normalize(x,m,s):

    return (x-m)/s



def normalize_to(data, train):

    mean, std = train.mean(), train.std()

    return normalize(data, mean, std), normalize(train, mean, std)



def get_normalized_data():

    X_train, y_train = get_data(path, 'train.csv')

    X_test, y_test = get_data(path,'test.csv')

    X_test, X_train = normalize_to(X_test, X_train)

    return X_train, y_train, X_test, y_test



def get_stats(x):

    return f'mean :{x.mean()} , std : {x.std()}';



def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_preds(model,dl):

    model.eval()

    preds = []

    for xb,yb in dl:

        xb,yb = xb.cuda(), yb.cuda()

        pred= model(xb)

        preds += pred.cpu().detach()

    return preds
class GeneralReLU(nn.Module):

    """Implementation of ReLU with added features like max_value, subtracting value, and leak.



    """

    def __init__(self, leak=None, sub=None, maxv=None):

        super().__init__()

        self.leak, self.sub, self.maxv = leak, sub, maxv;



    def forward(self, x):

        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)

        if self.sub is not None: x.sub_(self.sub);

        if self.maxv is not None: x.clamp_max_(self.maxv)

        return x
def conv_layer(f_in, f_out, ks, s, p):

    return nn.Sequential(nn.Conv2d(f_in, f_out, kernel_size=ks, stride=s, padding=p,bias=False),

          nn.BatchNorm2d(f_out),

          GeneralReLU(sub=0.5))

class ResBlock(nn.Module):

    def __init__(self, nf):

        super().__init__()

        self.nf = nf

        self.conv1 = conv_layer(nf,nf,3,1,1)

        self.conv2 = conv_layer(nf,nf,3,1,1)



    def forward(self, X):

        return X + self.conv2(self.conv1(X))
class DenseBlock(nn.Module):

    def __init__(self, ni, nf):

        super().__init__()

        self.ni, self.nf = ni, nf

        self.conv1 = conv_layer(ni, nf,3,1,1)

        self.conv2 = conv_layer(nf, nf,3,1,1)



    def forward(self, X):

        return torch.cat([X,self.conv2(self.conv1(X))],dim=1)
layers = nn.Sequential(Lambda(mnist_resize),

                       conv_layer(1,8,5,1,2), #14

                       nn.Dropout2d(p=0.05),

                       ResBlock(8),

                       nn.Dropout2d(p=0.05),

                       nn.MaxPool2d(3,2,1), #7

                       DenseBlock(8,8),

                       nn.Dropout2d(p=0.05),

                       nn.MaxPool2d(3,2,1), #4

                       DenseBlock(16,16),

                       nn.Dropout2d(p=0.05),

                       nn.AdaptiveAvgPool2d(1),

                       Lambda(flatten),

                       nn.Linear(32,10),

                       nn.BatchNorm1d(10)

                       )
X_train, y_train, X_test, y_test = get_normalized_data()

train_dl, valid_dl = get_dls(Dataset(X_train,y_train), Dataset(X_test,y_test))

model = get_model(layers=layers)

opt = get_optimizer(model)

loss_func = nn.CrossEntropyLoss()

init_cnn(model)
count_parameters(model)
one_cycle_sched= combine_scheds([0.3,0.7], [sched_cos(1e-3,1e-1), sched_cos(0.1,1e-6)])

fit_one_cycle(30,one_cycle_sched)
preds = get_preds(model,valid_dl)

res = []

for t in preds:

    r = t.argmax().item()

    res.append(r)
submission = pd.read_csv(path/'sample_submission.csv')

submission['Label'] = res

submission.to_csv('subs.csv',index=False)