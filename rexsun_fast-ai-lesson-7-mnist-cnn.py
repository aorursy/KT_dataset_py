%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
path=untar_data(URLs.MNIST)
path.ls()
il=ImageList.from_folder(path,convert_mode="L")
il.items[0]
defaults.camp="binary"
il
il[0].show()
sd=il.split_by_folder(train="training",valid="testing")
sd
(path/"training").ls()
l1=sd.label_from_folder()
l1
x,y=l1.train[0]
x.show()

print(y,x.shape)
tfms=([*rand_pad(padding=3,size=28,mode="zeros")],[])
l1=l1.transform(tfms)
bs=128
data=l1.databunch(bs=bs).normalize()
data
x,y=data.train_ds[0]

x.show()

print(y,x.shape)
# 画出一组经过数据变换后的图片

def _plot(i,j,ax): data.train_ds[0][0].show(ax,camp="gray")

plot_multi(_plot,3,3,figsize=(8,8))
xb,yb=data.one_batch()

xb.shape,yb.shape
# 展示一组样本

data.show_batch(rows=3,figsize=(5,5))
# 为了防止重复的写代码

def conv(ni,nf): return nn.Conv2d(ni,nf,kernel_size=3,stride=2,padding=1)
model=nn.Sequential(

    conv(1,8),

    nn.BatchNorm2d(8),

    nn.ReLU(),

    conv(8,16),

    nn.BatchNorm2d(16),

    nn.ReLU(),

    conv(16,32),

    nn.BatchNorm2d(32),

    nn.ReLU(),

    conv(32,16),

    nn.BatchNorm2d(16),

    nn.ReLU(),

    conv(16,10),

    nn.BatchNorm2d(10),

    Flatten()     # 扁平化

)
learn=Learner(data,model,loss_func=nn.CrossEntropyLoss(),metrics=accuracy)
learn.summary()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3,max_lr=0.1)
# fastai整合了一个conv、batchnorm和relu的函数

def conv(ni,nf): return conv_layer(ni,nf,stride=2)
model=nn.Sequential(

    conv(1,8),

    conv(8,16),

    conv(16,32),

    conv(32,16),

    conv(16,10),

    Flatten()

)
learn=Learner(data,model,loss_func=nn.CrossEntropyLoss(),metrics=accuracy)
learn.summary()
learn.fit_one_cycle(10,max_lr=0.1)
class ResBlock(nn.Module):

    def __init__(self,nf):

        super().__init__()

        self.conv1=conv_layer(nf,nf)

        self.conv2=conv_layer(nf,nf)

        

    def forward(self,x):

        return x+self.conv2(self.conv1(x))
model=nn.Sequential(

    conv(1,8),

    res_block(8),

    conv(8,16),

    res_block(16),

    conv(16,32),

    res_block(32),

    conv(32,16),

    res_block(16),

    conv(16,10),

    Flatten()

)
# 我们为了避免重复写代码

def conv_and_res(ni,nf): return nn.Sequential(conv(ni,nf),res_block(nf))
model=nn.Sequential(

    conv_and_res(1,8),

    conv_and_res(8,16),

    conv_and_res(16,32),

    conv_and_res(32,16),

    conv(16,10),

    Flatten()

)
learn=Learner(data,model,loss_func=nn.CrossEntropyLoss(),metrics=accuracy)
learn.summary()
learn.lr_find(end_lr=100)

learn.recorder.plot()
learn.fit_one_cycle(12,max_lr=0.1)