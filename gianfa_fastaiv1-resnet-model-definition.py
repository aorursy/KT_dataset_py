%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai.vision import *
!ls ../input
path = untar_data(URLs.MNIST)

path.ls()
il = ImageList.from_folder(path, convert_mode='L')

il
print(il.items[0])

defaults.cmap='binary' # <-- try to comment if you want to see a different colormap

il[0].show() # <- you can access the item by index
sd = il.split_by_folder(train='training', valid='testing')

sd
(path/'training').ls()
ll = sd.label_from_folder()

ll
x,y = ll.train[0] # we can access both the variables in Train set

print(y)

x
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])

ll = ll.transform(tfms)

ll
bs = 128

data = ll.databunch(bs=bs).normalize()
x,y = data.train_ds[0]

print(y)

x
def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')

plot_multi(_plot, 3, 3, figsize=(8,8))
xb, yb = data.one_batch()

print(xb.shape, yb.shape)

data.show_batch(rows=3, figsize=(5,5))
def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
nInput = 1

nOutput = 10 # <-- number of classes



model = nn.Sequential(

    conv(nInput, 8), # 14  (input, output)  <--- just defined 'conv' function

    nn.BatchNorm2d(8), # (input=prev_output)

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



    conv(16, nOutput), # 1   (input, OUTPUT)

    nn.BatchNorm2d(10), #

    Flatten()     # remove (1,1) grid

)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

print(learn.summary())
xb = xb.cuda()

xb
model(xb).shape
learn.lr_find(end_lr=100)

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(3, max_lr=4.5*0.1)
def conv2(ni, nf): return conv_layer(ni, nf, stride=2)
nInput = 1

nOutput = 10 

model = nn.Sequential(

    conv2(nInput, 8),   # 14

    conv2(8, 16),  # 7

    conv2(16, 32), # 4

    conv2(32, 16), # 2

    conv2(16, nOutput), # 1

    Flatten()      # remove (1,1) grid

)

learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
learn.fit_one_cycle(10, max_lr=0.1)

learn.record.plot_lr()
class ResBlock(nn.Module):

    def __init__(self, nf):

        super().__init__()

        self.conv1 = conv_layer(nf,nf)

        self.conv2 = conv_layer(nf,nf)

        

    def forward(self, x): return x + self.conv2(self.conv1(x))
help(res_block)
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