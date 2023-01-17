#!conda update -c pytorch -c fastai fastai
import fastai

import fastai.utils.collect_env

fastai.utils.collect_env.show_install(1)
%matplotlib inline

from fastai import *

from fastai.vision import *
path = untar_data(URLs.MNIST)
path.ls()
il = ImageItemList.from_folder(path, convert_mode='L')
il.items[0]
defaults.cmap='binary'
il
il[0].show()
sd = il.split_by_folder(train='training', valid='testing')
sd
(path/'training').ls()
ll = sd.label_from_folder()
ll
x,y = ll.train[0]
x.show()

print(y,x.shape)
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
ll = ll.transform(tfms)
bs = 128
# not using imagenet_stats because not using pretrained model

data = ll.databunch(bs=bs).normalize()
x,y = data.train_ds[0]
x.show()

print(y)
def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')

plot_multi(_plot, 3, 3, figsize=(8,8))
xb,yb = data.one_batch()

xb.shape,yb.shape
data.show_batch(rows=3, figsize=(5,5))
def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
model = nn.Sequential(

    conv(1, 8), # 14

    nn.BatchNorm2d(8),

    nn.ReLU(),

    conv(8, 16), # 7

    nn.BatchNorm2d(16),

    nn.ReLU(),

    conv(16, 32), # 4

    nn.BatchNorm2d(32),

    nn.ReLU(),

    conv(32, 16), # 2

    nn.BatchNorm2d(16),

    nn.ReLU(),

    conv(16, 10), # 1

    nn.BatchNorm2d(10),

    Flatten()     # remove (1,1) grid

)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
learn.summary()
xb = xb.cuda()
model(xb).shape
learn.lr_find(end_lr=100)
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=0.1)
def conv2(ni,nf): return conv_layer(ni,nf,stride=2)
model = nn.Sequential(

    conv2(1, 8),   # 14

    conv2(8, 16),  # 7

    conv2(16, 32), # 4

    conv2(32, 16), # 2

    conv2(16, 10), # 1

    Flatten()      # remove (1,1) grid

)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
learn.fit_one_cycle(10, max_lr=0.1)
from fastai.layers import *
class ResBlock(nn.Module):

    def __init__(self, nf):

        super().__init__()

        self.conv1 = conv_layer(nf,nf)

        self.conv2 = conv_layer(nf,nf)

        

    def forward(self, x): return x + self.conv2(self.conv1(x))
help(res_block)
class SequentialEx(nn.Module):

    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"

    def __init__(self, *layers):

        super().__init__()

        self.layers = nn.ModuleList(layers)



    def forward(self, x):

        res = x

        for l in self.layers:

            res.orig = x

            nres = l(res)

            # We have to remove res.orig to avoid hanging refs and therefore memory leaks

            res.orig = None

            res = nres

        return res



    def __getitem__(self,i): return self.layers[i]

    def append(self,l): return self.layers.append(l)

    def extend(self,l): return self.layers.extend(l)

    def insert(self,i,l): return self.layers.insert(i,l)
class MergeLayer(nn.Module):

    "Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`."

    def __init__(self, dense:bool=False):

        super().__init__()

        self.dense=dense



    def forward(self, x): return torch.cat([x,x.orig], dim=1) if self.dense else (x+x.orig)
def res_block(nf, dense:bool=False, norm_type:Optional[NormType]=NormType.Batch, bottle:bool=False, **kwargs):

    "Resnet block of `nf` features."

    norm2 = norm_type

    if not dense and (norm_type==NormType.Batch): norm2 = NormType.BatchZero

    nf_inner = nf//2 if bottle else nf

    return SequentialEx(conv_layer(nf, nf_inner, norm_type=norm_type, **kwargs),

                      conv_layer(nf_inner, nf, norm_type=norm2, **kwargs),

                      MergeLayer(dense))
model = nn.Sequential(

    conv2(1, 8),

    res_block(8),

    conv2(8, 16),

    res_block(16),

    conv2(16, 32),

    res_block(32),

    conv2(32, 16),

    res_block(16),

    conv2(16, 10),

    Flatten()

)
def conv_and_res(ni,nf): return nn.Sequential(conv2(ni, nf), res_block(nf))
model = nn.Sequential(

    conv_and_res(1, 8),

    conv_and_res(8, 16),

    conv_and_res(16, 32),

    conv_and_res(32, 16),

    conv2(16, 10),

    Flatten()

)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
learn.lr_find(end_lr=100)

learn.recorder.plot()
learn.fit_one_cycle(12, max_lr=0.05)
learn.summary()