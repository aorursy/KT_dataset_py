!pip install dataclasses
# Imports from the previous notebook
import pickle, gzip, torch, math, numpy as np, torch.nn.functional as F
from pathlib import Path
from IPython.core.debugger import set_trace
from dataclasses import dataclass
from typing import Any, Collection, Callable, NewType, List, Union, TypeVar, Optional
from functools import partial, reduce
from numbers import Number

from numpy import array
from torch import nn, optim, tensor, Tensor
from torch.utils.data import TensorDataset, Dataset, DataLoader
# Data download
from pathlib import Path
import requests

DATA_PATH = Path('data')
PATH = DATA_PATH/'mnist'

PATH.mkdir(parents=True, exist_ok=True)

URL='http://deeplearning.net/data/mnist/'
FILENAME='mnist.pkl.gz'

if not (PATH/FILENAME).exists():
    content = requests.get(URL+FILENAME).content
    (PATH/FILENAME).open('wb').write(content)
# Load the data and convert to tensors
with gzip.open(PATH/'mnist.pkl.gz', 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))
x_train.min(),x_train.max()
x_train
idx = x_train[0].nonzero()[0]
x_train[0][idx-3:idx+15]
%matplotlib inline

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28,28)), cmap="gray")
x_train.shape, y_train[0]
bs=64
epochs=2
lr=0.2
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
# Type definitions
Rank0Tensor = NewType('OneEltTensor', Tensor)
LossFunction = Callable[[Tensor, Tensor], Rank0Tensor]
Model = nn.Module
def is_listy(x:Any)->bool: return isinstance(x, (tuple,list))
def loss_batch(model:Model, xb:Tensor, yb:Tensor, 
               loss_fn:LossFunction, opt:optim.Optimizer=None):
    "Calculate loss for the batch `xb,yb` and backprop with `opt`"
    loss = loss_fn(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return loss.item(), len(yb)
def fit(epochs:int, model:Model, loss_fn:LossFunction, 
        opt:optim.Optimizer, train_dl:DataLoader, valid_dl:DataLoader):
    "Train `model` on `train_dl` with `optim` then validate against `valid_dl`"
    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl: loss,_ = loss_batch(model, xb, yb, loss_fn, opt)

        model.eval()
        with torch.no_grad():
            losses,nums = zip(*[loss_batch(model, xb, yb, loss_fn)
                                for xb,yb in valid_dl])
        val_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)

        print(epoch, val_loss)
LambdaFunc = Callable[[Tensor],Tensor]
class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`"
    def __init__(self, func:LambdaFunc):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func
        
    def forward(self, x): return self.func(x)
#export
def noop(x): return x

def ResizeBatch(*size:int) -> Tensor: 
    "Layer that resizes x to `size`, good for connecting mismatched layers"
    return Lambda(lambda x: x.view((-1,)+size))

def Flatten()->Tensor: 
    "Flattens `x` to a single dimension, often used at the end of a model"
    return Lambda(lambda x: x.view((x.size(0), -1)))

def PoolFlatten()->nn.Sequential:
    "Apply `nn.AdaptiveAvgPool2d` to `x` and then flatten the result"
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten())

def conv2d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias=False) -> nn.Conv2d:
    "Create `nn.Conv2d` layer: `ni` inputs, `nf` outputs, `ks` kernel size. `padding` defaults to `k//2`"
    if padding is None: padding = ks//2
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)

def conv2d_relu(ni:int, nf:int, ks:int=3, stride:int=1, 
                padding:int=None, bn:bool=False) -> nn.Sequential:
    "Create a `conv2d` layer with `nn.ReLU` activation and optional(`bn`) `nn.BatchNorm2d`"
    layers = [conv2d(ni, nf, ks=ks, stride=stride, padding=padding), nn.ReLU()]
    if bn: layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)

def conv2d_trans(ni:int, nf:int, ks:int=2, stride:int=2, padding:int=0) -> nn.ConvTranspose2d:
    "Create `nn.nn.ConvTranspose2d` layer: `ni` inputs, `nf` outputs, `ks` kernel size. `padding` defaults to 0"
    return nn.ConvTranspose2d(ni, nf, kernel_size=ks, stride=stride, padding=padding)
model = nn.Sequential(
    ResizeBatch(1,28,28),
    conv2d_relu(1,  16), 
    conv2d_relu(16, 16),
    conv2d_relu(16, 10),
    PoolFlatten()
)
def get_data(train_ds, valid_ds, bs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(valid_ds, batch_size=bs*2))

train_dl,valid_dl = get_data(train_ds, valid_ds, bs)
loss_fn = F.cross_entropy
opt = optim.SGD(model.parameters(), lr=lr)
loss_fn(model(x_valid[0:bs]), y_valid[0:bs])
fit(epochs, model, loss_fn, opt, train_dl, valid_dl)
def mnist2image(b): return b.view(1,28,28)
@dataclass
class DatasetTfm(Dataset):
    "Applies `tfm` to `ds`"
    ds: Dataset
    tfm: Callable = None
        
    def __len__(self): return len(self.ds)
    
    def __getitem__(self,idx:int):
        "Apply `tfm` to `x` and return `(x[idx],y[idx])`"
        x,y = self.ds[idx]
        if self.tfm is not None: x = self.tfm(x)
        return x,y
train_tds = DatasetTfm(train_ds, mnist2image)
valid_tds = DatasetTfm(valid_ds, mnist2image)
def get_data(train_ds, valid_ds, bs):
    return (DataLoader(train_ds, bs,   shuffle=True),
            DataLoader(valid_ds, bs*2, shuffle=False))
train_dl,valid_dl = get_data(train_tds, valid_tds, bs)
x,y = next(iter(valid_dl))
valid_ds[0][0].shape, x[0].shape
torch.allclose(valid_ds[0][0], x[0].view(-1))
def simple_cnn(channels:Collection[int], kernel_szs:Collection[int], 
               strides:Collection[int]) -> nn.Sequential:
    "CNN with `conv2d_relu` layers defined by `actns`, `kernel_szs` and `strides`"
    layers = [conv2d_relu(channels[i], channels[i+1], kernel_szs[i], stride=strides[i])
        for i in range(len(strides))]
    layers.append(PoolFlatten())
    return nn.Sequential(*layers)
def get_model():
    model = simple_cnn([1,16,16,10], [3,3,3], [2,2,2])
    return model, optim.SGD(model.parameters(), lr=lr)
model,opt = get_model()
model
fit(epochs, model, loss_fn, opt, train_dl, valid_dl)
def ifnone(a:bool,b:Any):
    "`a` if its not None, otherwise `b`"
    return b if a is None else a
default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Tensors = Union[Tensor, Collection['Tensors']]
def to_device(b:Tensors, device:torch.device):
    "Ensure `b` is on `device`"
    device = ifnone(device, default_device)
    if is_listy(b): return [to_device(o, device) for o in b]
    return b.to(device)
@dataclass
class DeviceDataLoader():
    "`DataLoader` that ensures batches from `dl` are on `device`"
    dl: DataLoader
    device: torch.device

    def __len__(self) -> int: return len(self.dl)
    def proc_batch(self,b:Tensors): return to_device(b, self.device)

    def __iter__(self)->Tensors:
        "Ensure batches from `dl` are on `device` as we iterate"
        self.gen = map(self.proc_batch, self.dl)
        return iter(self.gen)

    @classmethod
    def create(cls, *args, device:torch.device=default_device, **kwargs): 
        return cls(DataLoader(*args, **kwargs), device=device)
def get_data(train_ds, valid_ds, bs):
    return (DeviceDataLoader.create(train_ds, bs,   shuffle=True,  num_workers=2),
            DeviceDataLoader.create(valid_ds, bs*2, shuffle=False, num_workers=2))
train_dl,valid_dl = get_data(train_tds, valid_tds, bs)
def get_model():
    model = simple_cnn([1,16,16,10], [3,3,3], [2,2,2]).to(default_device)
    return model, optim.SGD(model.parameters(), lr=lr)
model,opt = get_model()
x,y = next(iter(valid_dl))
x.type(),y.type()
def fit(epochs:int, model:Model, loss_fn:LossFunction, 
        opt:optim.Optimizer, train_dl:DataLoader, valid_dl:DataLoader) -> None:
    "Train `model` for `epochs` with `loss_fun` and `optim`"
    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl: loss,_ = loss_batch(model, xb, yb, loss_fn, opt)

        model.eval()
        with torch.no_grad():
            losses,nums = zip(*[loss_batch(model, xb, yb, loss_fn)
                                for xb,yb in valid_dl])
        val_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)

        print(epoch, val_loss)
fit(5, model, loss_fn, opt, train_dl, valid_dl)
TItem = TypeVar('TItem')
TfmCallable = Callable[[TItem],TItem]
TfmList = Union[TfmCallable, Collection[TfmCallable]]
Tfms = Optional[TfmList]
@dataclass
class DataBunch():
    "Bind `train_dl`, `valid_dl` to `device`"
    train_dl:DataLoader
    valid_dl:DataLoader
    device:torch.device=None

    @classmethod
    def create(cls, train_ds:Dataset, valid_ds:Dataset, bs:int=64, 
               train_tfm:Tfms=None, valid_tfm:Tfms=None, device:torch.device=None, **kwargs):
        return cls(DeviceDataLoader.create(DatasetTfm(train_ds, train_tfm), bs,   shuffle=True,  device=device, **kwargs),
                   DeviceDataLoader.create(DatasetTfm(valid_ds, valid_tfm), bs*2, shuffle=False, device=device, **kwargs),
                   device=device)
class Learner():
    "Train `model` on `data` for `epochs` using learning rate `lr` and `opt_fn` to optimize training"
    def __init__(self, data:DataBunch, model:Model):
        self.data,self.model = data,to_device(model, data.device)

    def fit(self, epochs, lr, opt_fn=optim.SGD, loss_fn=F.cross_entropy):
        opt = opt_fn(self.model.parameters(), lr=lr)
        fit(epochs, self.model, loss_fn, opt, self.data.train_dl, self.data.valid_dl)
data = DataBunch.create(train_ds, valid_ds, bs=bs, train_tfm=mnist2image, valid_tfm=mnist2image)
model = simple_cnn([1,16,16,10], [3,3,3], [2,2,2])
learner = Learner(data, model)
opt_fn = partial(optim.SGD, momentum=0.9)
learner.fit(1, lr/5, opt_fn=opt_fn)
learner.fit(2, lr, opt_fn=opt_fn)
learner.fit(1, lr/5, opt_fn=opt_fn)
