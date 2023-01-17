import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
DATA_DIR = '../input/'
W, H = 9, 9
# train = pd.read_csv(DATA_DIR+'train_{}_{}_mat.csv'.format(W, H))
# test = pd.read_csv(DATA_DIR+'test_{}_{}_mat.csv'.format(W, H))
train = pd.read_csv(DATA_DIR+'train.csv'.format(W, H))
test = pd.read_csv(DATA_DIR+'test.csv'.format(W, H))
X = train.iloc[:, 1:].values
y = train.iloc[:, 0].values
X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

X = X / X.max().max().astype(np.float32)
X_test = X_test / X_test.max().max().astype(np.float32)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=42)
import time
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
class LinearModel(nn.Module):
    def __init__(self, dim_input, n_class):
        super(LinearModel, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        torch.manual_seed(233)
        self.dim_input = dim_input
        self.linear_model = nn.Linear(dim_input, n_class, bias=False)
    def eval_metric(self, y_true, y_pred):
        y_pred_ = np.array(y_pred)
        y_true_ = np.array(y_true)
        if len(y_pred_.shape)>1 and y_pred_.shape[1]>1:
            return accuracy_score(y_true_, np.argmax(y_pred_, -1))      
    def forward(self, x):
        x_out = self.linear_model(x)
        return F.log_softmax(x_out, dim=-1)
# batch_size = 2000
n_epochs = 1000
eval_results = {
    'train': np.zeros(n_epochs), 
    'valid': np.zeros(n_epochs)
}
patience = 5
optim_params = dict(lr=0.03, weight_decay=1e-6)
n_class = 10
n_feature = X_train.shape[1]
verbose_eval = 50
use_cuda = torch.cuda.is_available()
model = LinearModel(n_feature, n_class)
if use_cuda:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), **optim_params)
criterion = F.cross_entropy
eval_metric = accuracy_score
dtrain = Variable(torch.from_numpy(X_train).float())
ltrain = Variable(torch.from_numpy(y_train).long())
dvalid = Variable(torch.from_numpy(X_valid).float())
lvalid = Variable(torch.from_numpy(y_valid).long())
dtest = Variable(torch.from_numpy(X_test).float())
ltest = Variable(torch.from_numpy(y_test).long())
if use_cuda:
    dtrain = dtrain.cuda()
    ltrain = ltrain.cuda()
    dvalid = dvalid.cuda()
    lvalid = lvalid.cuda()
    dtest = dtest.cuda()
    ltest = ltest.cuda()
for epoch_i in np.arange(n_epochs):
    def closure():
        model.train()
        optimizer.zero_grad()
        out = model(dtrain)
        loss = criterion(out, ltrain)
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    eval_results['train'][epoch_i] = float(loss)
    model.eval()
    out = model(dvalid)
    eval_results['valid'][epoch_i] = float(criterion(out, lvalid))
    if use_cuda:
        out = out.cpu()
    y_pred = np.argmax(out.data.numpy(), -1)
    if (epoch_i+1) % verbose_eval == 0:
        print(f"loss for {epoch_i+1} epoch : {float(loss):.6f}" )
        print(f'valid metric {eval_metric(y_valid, y_pred):.6f}')
ax = pd.DataFrame(eval_results).plot(figsize=[10, 6], grid=1)

model.eval()
out = model(dtest)
if use_cuda:
    out = out.cpu()
y_pred = np.argmax(out.data.numpy(), -1)
print(f'test metric {eval_metric(y_test, y_pred):.6f}')