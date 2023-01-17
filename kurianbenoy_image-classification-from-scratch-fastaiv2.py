#Run once per session

!pip install fastai2
# grab vision related APIs

from fastai2.basics import *

from fastai2.vision.all import *

from fastai2.callback.all import *
# Download our data



path = untar_data(URLs.MNIST)
items= get_image_files(path)

items[0]
# create an image object with ImageBlock



img = PILImageBW.create(items[0])

img.show()
# Split our data with GrandparentSplitter, which will make use of a train and valid folder.



splits = GrandparentSplitter(train_name='training', valid_name='testing')

splits = splits(items)
# understand what it's split with?

splits[0][:5], splits[1][:5]
dsrc = Datasets(items, tfms=[[PILImageBW.create],[parent_label, Categorize]],

                   splits=splits)
show_at(dsrc.train, 3)
tfms = [ToTensor(), CropPad(size=34, pad_mode=PadMode.Zeros), RandomCrop(size=28)]

gpu_tfms = [IntToFloatTensor(), Normalize()]


dls = dsrc.dataloaders(bs=128, after_item=tfms, after_batch=gpu_tfms)
dls.show_batch()
# passing as a batch

xb, yb = dls.one_batch()

xb.shape, yb.shape
# no of classes and class labels

dls.vocab
def conv(ni, nf):

    return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
def bn(nf): return nn.BatchNorm2d(nf)
def ReLU(): return nn.ReLU(inplace=True)
# five CNN layers, 1 -> 32 -> 10



model = nn.Sequential(

            conv(1,8),

            bn(8),

            ReLU(),

            conv(8,16),

            bn(16),

            ReLU(),

            conv(16, 32),

            bn(32),

            ReLU(),

            conv(32,16),

            bn(16),

            ReLU(),

            conv(16,10),

            bn(10),

            Flatten() # and flatten it into a single dimention of predictions

)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

learn.summary()
learn.lr_find()
learn.fit_one_cycle(3, lr_max=1e-1)
def conv2(ni, nf):

    return ConvLayer(ni, nf, stride=2)
net = nn.Sequential(conv2(1,8),

                   conv2(8,16),

                   conv2(16,32),

                   conv2(32,16),

                   conv2(16,10),

                   Flatten())
learn = Learner(dls, net, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(3, lr_max=1e-1)
class ResBlock(Module):

    def __init__(self, nf):

        self.conv1 = ConvLayer(nf, nf)

        self.conv2 = ConvLayer(nf, nf)

  

    def forward(self, x): return x + self.conv2(self.conv1(x))
net = nn.Sequential(

    conv2(1,8),

    ResBlock(8),

    conv2(8,16),

    ResBlock(16),

    conv2(16,32),

    ResBlock(32),

    conv2(32,16),

    ResBlock(16),

    conv2(16,10),

    Flatten()

)
net
def conv_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), ResBlock(nf))
net = nn.Sequential(

    conv_and_res(1,8),

    conv_and_res(8,16),

    conv_and_res(16,32),

    conv_and_res(32,16),

    conv2(16,10),

    Flatten()

)
learn = Learner(dls, net, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

learn.fit_one_cycle(3, lr_max=1e-1)