import pandas as pd

import matplotlib.pyplot as plt



import numpy as np

import os

from sklearn.metrics import f1_score



from fastai import *

from fastai.vision import *



import torch

import torch.nn as nn

import torchvision

import cv2



from tqdm import tqdm

from skmultilearn.model_selection import iterative_train_test_split

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer

import warnings

warnings.filterwarnings("ignore")



%load_ext autoreload

%autoreload
!git clone https://github.com/adambielski/GrouPy.git



import sys

sys.path.insert(0,'./GrouPy/')



import groupy

import torch

import torch.nn as nn

import torch.nn.functional as F



from torch.autograd import Variable

from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4

from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling





from torch.autograd import Variable

from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()

        self.conv1 = P4MConvP4M(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = P4MConvP4M(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm3d(planes)



        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut = nn.Sequential(

                P4MConvP4M(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm3d(self.expansion*planes)

            )



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)

        out = F.relu(out)

        return out





class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, in_planes, planes, stride=1):

        super(Bottleneck, self).__init__()

        self.conv1 = P4MConvP4M(in_planes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = P4MConvP4M(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = P4MConvP4M(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm3d(self.expansion*planes)



        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut = nn.Sequential(

                P4MConvP4M(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm3d(self.expansion*planes)

            )



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)

        out = F.relu(out)

        return out





class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):

        super(ResNet, self).__init__()

        self.in_planes = 23



        self.conv1 = P4MConvZ2(3, 23, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm3d(23)

        self.layer1 = self._make_layer(block, 23, num_blocks[0], stride=1)

        self.layer2 = self._make_layer(block, 45, num_blocks[1], stride=2)

        self.layer3 = self._make_layer(block, 91, num_blocks[2], stride=2)

        self.layer4 = self._make_layer(block, 181, num_blocks[3], stride=2)

        self.linear = nn.Linear(181*8*block.expansion, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)

        layers = []

        for stride in strides:

            layers.append(block(self.in_planes, planes, stride))

            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)

        

        out = self.layer2(out)

        

        out = self.layer3(out)

        

        out = self.layer4(out)

        

        outs = out.size()

        print(out.size(),out.size(0))

        

        out = out.view(outs[0], outs[1]*outs[2], outs[3], outs[4])

        print(out.size(),out.size(0))



        out = F.avg_pool2d(out, 4)

        print(out.size(),out.size(0))

        out = out.view(out.size(0), -1)

        print(out.size())

        out = self.linear(out)

        print(out.size())

        return out





def ResNet18(pretrained=False):

    return ResNet(BasicBlock, [2,2,2,2])



def ResNet34(pretrained=False):

    return ResNet(BasicBlock, [3,4,6,3])



def ResNet50(pretrained=False):

    return ResNet(Bottleneck, [3,4,6,3])



def ResNet101(pretrained=False):

    return ResNet(Bottleneck, [3,4,23,3])



def ResNet152(pretrained=False):

    return ResNet(Bottleneck, [3,8,36,3])
model_path='.'

path='../input/'

train_folder=f'{path}train'

test_folder=f'{path}test'

train_lbl=f'{path}train_labels.csv'

ORG_SIZE=96



bs=32

num_workers=None # Apprently 2 cpus per kaggle node, so 4 threads I think

sz=96
df_trn=pd.read_csv(train_lbl)
tfms = get_transforms(do_flip=False, flip_vert=False, max_rotate=.0, max_zoom=.3,

                      max_lighting=0.15, max_warp=0.10)
data = ImageDataBunch.from_csv(path,csv_labels=train_lbl,folder='train',bs=bs, ds_tfms=tfms, size=sz, suffix='.tif',test=test_folder);

stats=data.batch_stats()        

data.normalize(stats)
data.show_batch(rows=3, figsize=(12,9))
from sklearn.metrics import roc_auc_score

def auc_score(y_pred,y_true):  

    return roc_auc_score(to_np(y_true),to_np(F.sigmoid(y_pred))[:,1])
#f1=Fbeta_binary(beta2=1)

f1=partial(fbeta,beta=1)
class AdaptiveConcatPool3d(nn.Module):

    "Layer that concats `AdaptiveAvgPool3d` and `AdaptiveMaxPool3d`."

    def __init__(self, sz:Optional[int]=None):

        "Output will be 2*sz or 2 if sz is None"

        super().__init__()

        sz = sz or 1

        self.ap,self.mp = nn.AdaptiveAvgPool3d(sz), nn.AdaptiveMaxPool3d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
def create_head_custom(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False):

    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."

    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]

    ps = listify(ps)

    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps

    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]

    layers = [AdaptiveConcatPool3d(), Flatten()]

    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):

        layers += bn_drop_lin(ni,no,True,p,actn)

    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))

    return nn.Sequential(*layers)

learn = create_cnn(

    data,

    ResNet18,

    path='.',    

    metrics=[fbeta], 

    ps=0.5,

    pretrained=False

)
learn.model[1]=create_head_custom(362,2)

learn.model.to(learn.data.device)

print(learn.summary())

learn.lr_find(num_it=500,end_lr=1000)

learn.recorder.plot()
learn.fit_one_cycle(2,slice(5e-5,1e-4))

learn.save('gCNN_1cycle')
learn.recorder.plot()

try:

    learn.recorder.plot_losses()

    learn.recorder.plot_lr()

except:

    pass
preds,y=learn.get_preds()

pred_score=auc_score(preds,y)

pred_score
preds_test,y_test=learn.get_preds(ds_type=DatasetType.Test)
preds_test_tta,y_test_tta=learn.TTA(ds_type=DatasetType.Test)
sub=pd.read_csv(f'{path}/sample_submission.csv').set_index('id')

sub.head()
clean_fname=np.vectorize(lambda fname: str(fname).split('/')[-1].split('.')[0])

fname_cleaned=clean_fname(data.test_ds.items)

fname_cleaned=fname_cleaned.astype(str)
sub.loc[fname_cleaned,'label']=to_np(preds_test[:,1])

sub.to_csv(f'submission_{pred_score}.csv')
sub.loc[fname_cleaned,'label']=to_np(preds_test_tta[:,1])

sub.to_csv(f'submission_TTA_{pred_score}.csv')
!rm -r GrouPy/