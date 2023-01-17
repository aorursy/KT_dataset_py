import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
DATA_DIR = '../input/'
W, H = 9, 9
train = pd.read_csv(DATA_DIR+'train_{}_{}_mat.csv'.format(W, H))
test = pd.read_csv(DATA_DIR+'test_{}_{}_mat.csv'.format(W, H))
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
n_epochs = 2000
eval_results = {
    'loss': np.zeros(n_epochs), 
    'val_loss': np.zeros(n_epochs),
    'acc': np.zeros(n_epochs), 
    'val_acc': np.zeros(n_epochs)
}
patience = 100
optim_params = dict(lr=0.03, weight_decay=1e-6)
n_class = 10
n_feature = X_train.shape[1]
verbose_eval = 100
save_path = 'model.pth'
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
restarted = 0
best_score = -1
best_epoch = -1
for epoch_i in np.arange(n_epochs):
    def closure():
        model.train()
        optimizer.zero_grad()
        out = model(dtrain)
        loss = criterion(out, ltrain)
        loss.backward()
        return loss, out
    loss, out = optimizer.step(closure)
    eval_results['loss'][epoch_i] = float(loss)
    if use_cuda:
        out = out.cpu()
    y_pred = np.argmax(out.data.numpy(), -1)
    eval_results['acc'][epoch_i] = eval_metric(y_train, y_pred)
    model.eval()
    out = model(dvalid)
    val_loss = float(criterion(out, lvalid))
    eval_results['val_loss'][epoch_i] = val_loss
    if use_cuda:
        out = out.cpu()
    y_pred = np.argmax(out.data.numpy(), -1)
    val_score = eval_metric(y_valid, y_pred)
    eval_results['val_acc'][epoch_i] = val_score
    if val_score>best_score:
        best_score = val_score
        best_epoch = epoch_i
        torch.save(model.state_dict(), save_path)
        restarted = 0
    else:
        restarted += 1
    if (epoch_i+1) % verbose_eval == 0:
        print(f"epoch {epoch_i+1} loss {val_loss:.6f}" )
        print(f'valid metric {val_score:.6f}')
model.load_state_dict(torch.load(save_path))
eval_res = pd.DataFrame(eval_results)
plot_params = dict(figsize=[12, 4], grid=True)
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120)
eval_res[['loss', 'val_loss']].iloc[200:].plot(ax=axes[0], **plot_params)
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
eval_res[['acc', 'val_acc']].iloc[200:].plot(ax=axes[1], **plot_params)
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('acc')
print(f'best score {best_score:.6f} @ epoch {best_epoch}')

model.eval()
out = model(dtest)
if use_cuda:
    out = out.cpu()
y_pred = np.argmax(out.data.numpy(), -1)
print(f'test metric {eval_metric(y_test, y_pred):.6f}')
f = plt.figure(dpi=120)
sns.distplot(model.linear_model.weight.cpu().data.numpy().ravel())
plt.grid()
plt.title('Weight distribution')
plt.show()