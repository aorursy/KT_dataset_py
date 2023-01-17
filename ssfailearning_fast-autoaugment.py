# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install hyperopt
!pip install geffnet
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
from PIL import ImageOps, ImageEnhance
from PIL import Image as I

class BaseTransform(ABC):

    def __init__(self, prob, mag):
        self.prob = prob
        self.mag = mag

    def __call__(self, img):
        return transforms.RandomApply([self.transform], self.prob)(img)

    def __repr__(self):
        return '%s(prob=%.2f, magnitude=%.2f)' % \
                (self.__class__.__name__, self.prob, self.mag)

    @abstractmethod
    def transform(self, img):
        pass


class ShearXY(BaseTransform):

    def transform(self, img):
        degrees = self.mag * 360
        t = transforms.RandomAffine(0, shear=degrees, resample=I.BILINEAR)
        return t(img)


class TranslateXY(BaseTransform):

    def transform(self, img):
        translate = (self.mag, self.mag)
        t = transforms.RandomAffine(0, translate=translate, resample=I.BILINEAR)
        return t(img)


class Rotate(BaseTransform):

    def transform(self, img):
        degrees = self.mag * 360
        t = transforms.RandomRotation(degrees, I.BILINEAR)
        return t(img)


class AutoContrast(BaseTransform):

    def transform(self, img):
        cutoff = int(self.mag * 49)
        return ImageOps.autocontrast(img, cutoff=cutoff)


class Invert(BaseTransform):

    def transform(self, img):
        return ImageOps.invert(img)


class Equalize(BaseTransform):

    def transform(self, img):
        return ImageOps.equalize(img)


class Solarize(BaseTransform):

    def transform(self, img):
        threshold = (1-self.mag) * 255
        return ImageOps.solarize(img, threshold)


class Posterize(BaseTransform):

    def transform(self, img):
        bits = int((1-self.mag) * 8)
        return ImageOps.posterize(img, bits=bits)


class Contrast(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Contrast(img).enhance(factor)


class Color(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Color(img).enhance(factor)


class Brightness(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Brightness(img).enhance(factor)


class Sharpness(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Sharpness(img).enhance(factor)


class Cutout(BaseTransform):

    def transform(self, img):
        n_holes = 1
        length = 24 * self.mag
        cutout_op = CutoutOp(n_holes=n_holes, length=length)
        return cutout_op(img)


class CutoutOp(object):
    """
    https://github.com/uoguelph-mlrg/Cutout
    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        w, h = img.size

        mask = np.ones((h, w, 1), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h).astype(int)
            y2 = np.clip(y + self.length // 2, 0, h).astype(int)
            x1 = np.clip(x - self.length // 2, 0, w).astype(int)
            x2 = np.clip(x + self.length // 2, 0, w).astype(int)

            mask[y1: y2, x1: x2, :] = 0.

        img = mask*np.asarray(img).astype(np.uint8)
        img = I.fromarray(mask*np.asarray(img))

        return img
import torchvision.transforms as transforms
import torch.utils.data as Data
from PIL import Image as I
import geffnet
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from fastai.vision import *
class TestDataset(Data.Dataset):
    def __init__(self, names, image_labels, transform):
        super(TestDataset, self).__init__()
        self.names = names
        self.image_labels = image_labels
        self.transform = transform

    def __getitem__(self, index):
        name = self.names[index]
        if type(name)==list:
            name=name[0]
        label = self.image_labels[name]
        image = I.open(name)
        image=self.transform(image)
        return image, label

    def __len__(self):
        return len(self.names)


DEFALUT_CANDIDATES = [
    ShearXY,
    TranslateXY,
    Rotate,
    AutoContrast,
    Invert,
    Equalize,
    Solarize,
    Posterize,
    Contrast,
    Color,
    Brightness,
    Sharpness,
    Cutout,
#     SamplePairing,
]


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        backbone = geffnet.efficientnet_b3(pretrained=True)
        self.backbone = torch.nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            backbone.act1,
            backbone.blocks,
            backbone.conv_head,
            backbone.bn2,
            backbone.act2,
            backbone.global_pool
        )

        self.global_avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.global_bn = nn.BatchNorm1d(1536)
        self.global_bn.bias.requires_grad = False
        self.local_conv = nn.Conv2d(1536, 512, 1)
        self.local_bn = nn.BatchNorm2d(512)
        self.local_bn.bias.requires_grad = False
        self.fc = nn.Linear(1536, 20)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.backbone(x)

        global_feat = self.global_avgpool(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        global_feat = F.dropout(global_feat, p=0.2)
        global_feat = self.global_bn(global_feat)
        global_feat = l2_norm(global_feat)

        local_feat = torch.mean(x, -1, keepdim=True)
        local_feat = self.local_bn(self.local_conv(local_feat))
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        local_feat = l2_norm(local_feat, axis=-1)

        out = self.fc(global_feat) * 16
        return global_feat, local_feat, out

import torch.nn as nn
import torch


class TripletLoss(nn.Module):

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def shortest_dist(self, dist_mat):
        m, n = dist_mat.size()[:2]
        dist = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if (i == 0) and (j == 0):
                    dist[i][j] = dist_mat[i, j]
                elif (i == 0) and (j > 0):
                    dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
                elif (i > 0) and (j == 0):
                    dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
                else:
                    dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
        dist = dist[-1][-1]
        return dist

    '''局部特征的距离矩阵'''

    def compute_local_dist(self, x, y):
        M, m, d = x.size()
        N, n, d = y.size()
        x = x.contiguous().view(M * m, d)
        y = y.contiguous().view(N * n, d)
        dist_mat = self.comput_dist(x, y)
        dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
        dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
        dist_mat = self.shortest_dist(dist_mat)
        return dist_mat

    '''全局特征的距离矩阵'''

    def comput_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def hard_example_mining(self, dist_mat, labels, return_inds=False):
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        list_ap = []
        list_an = []
        # print(is_neg)
        # print(dist_mat.shape)
        # print(dist_mat[1])
        # print(dist_mat[1][is_neg[1]])
        for i in range(N):
            list_ap.append(dist_mat[i][is_pos[i]].max().unsqueeze(0))
            list_an.append(dist_mat[i][is_neg[i]].min().unsqueeze(0))
        dist_ap = torch.cat(list_ap)
        dist_an = torch.cat(list_an)
        return dist_ap, dist_an

    def forward(self, feat_type, feat, labels):
        '''

        :param feat_type: 'global'代表计算全局特征的三重损失，'local'代表计算局部特征
        :param feat: 经过网络计算出来的结果
        :param labels: 标签
        :return:
        '''
        if feat_type == 'global':
            dist_mat = self.comput_dist(feat, feat)
        else:
            dist_mat = self.compute_local_dist(feat, feat)
        # print("dist_mat:{}".format(dist_mat))
        # print(dist_mat.shape)
        # print("labels:{}".format(labels))
        dist_ap, dist_an = self.hard_example_mining(
            dist_mat, labels)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


def one_hot_smooth_label(x, num_class, smooth=0.1):
    num = x.shape[0]
    labels = torch.zeros((num, 20))
    for i in range(num):
        labels[i][x[i]] = 1
    labels = (1 - (num_class - 1) / num_class * smooth) * labels + smooth / num_class
    return labels


class Criterion:
    def __init__(self):
        self.triplet_criterion = TripletLoss()
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def __call__(self, global_feat, local_feat, cls_score, label):
        global_loss = self.triplet_criterion('global', global_feat, label)
        local_loss = self.triplet_criterion('local', local_feat, label)
        label = one_hot_smooth_label(label, 20)
        cls_loss = self.cls_criterion(cls_score, label)

        return global_loss + local_loss + cls_loss


def validate_child(model,dl,criterion,transform):
    device=torch.device('cuda:0')
    model=model.to(device)
    model.eval()
    steps=len(dl)
    total_metric=0
    total_loss=0
    dl.dataset.transform=transform
    with torch.no_grad():
        for images,labels in dl:
            images=images.to(device)

            # print("labels:{}".format(labels))
            # print("images:{}".format(images))
            global_feat,local_feat,logits=model(images)
            # print("global_feat:{}".format(global_feat))
            # print("local_feat:{}".format(local_feat))
            global_feat = global_feat.to('cpu')
            logits = logits.to('cpu')
            local_feat = local_feat.to('cpu')
            loss=criterion(global_feat,local_feat,logits,labels)
            metric=accuracy(logits,labels)
            total_loss+=loss
            total_metric+=metric
    metric=total_metric/steps
    loss=total_loss/steps
    print('metric:{},loss:{}'.format(metric,loss))
    return metric,loss



def get_next_subpolicy(transform_candidates,op_per_subpolicy=2):
    n_candidates=len(transform_candidates)
    subpolicy=[]
    for i in range(op_per_subpolicy):
        index=random.randrange(n_candidates)
        prob=random.random()
        mag=random.random()
        subpolicy.append(transform_candidates[index](prob,mag))
    subpolicy=transforms.Compose([
        *subpolicy,
        transforms.Resize([300,300]),
        transforms.ToTensor()
    ])
    return subpolicy

def search_subpolicies_hyperopt(transform_candidates,child_model,dl,B,criterion):
    def _objective(sampled):
        subpolicy=[transform(prob,mag) for transform,prob,mag in sampled]
        subpolicy=transforms.Compose([
            transforms.Resize([300,300]),
            *subpolicy,
            transforms.ToTensor()
        ])

        val_res=validate_child(child_model,dl,criterion,subpolicy)
        loss=val_res[1].cpu().detach().numpy()
        return {'loss':loss,'status':STATUS_OK}

    space = [(hp.choice('transform1', transform_candidates), hp.uniform('prob1', 0, 1.0), hp.uniform('mag1', 0, 1.0)),
             (hp.choice('transform2', transform_candidates), hp.uniform('prob2', 0, 1.0), hp.uniform('mag2', 0, 1.0)),
             (hp.choice('transform3', transform_candidates), hp.uniform('prob3', 0, 1.0), hp.uniform('mag3', 0, 1.0))]
#              (hp.choice('transform4', transform_candidates), hp.uniform('prob4', 0, 1.0), hp.uniform('mag4', 0, 1.0))]

    trials = Trials()
    best = fmin(_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=B,
                trials=trials)
    subpolicies=[]
    for t in trials.trials:
        vals = t['misc']['vals']
        subpolicy = [transform_candidates[vals['transform1'][0]](vals['prob1'][0], vals['mag1'][0]),
                     transform_candidates[vals['transform2'][0]](vals['prob2'][0], vals['mag2'][0]),
                      transform_candidates[vals['transform3'][0]](vals['prob3'][0], vals['mag3'][0])]
        subpolicy = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            ## policy
            *subpolicy,
            ## to tensor
            transforms.ToTensor()])
        subpolicies.append((subpolicy, t['result']['loss']))
    return subpolicies

def get_topn_subpolicies(subpolicies,N=10):
    return sorted(subpolicies, key=lambda subpolicy: subpolicy[1])[:N]


def process_fn(child_model,Da_dl,T,transform_candidates,B,N):
    transform=[]


    criterion=Criterion()
    for i in range(T):
        subpolicies=search_subpolicies_hyperopt(transform_candidates,child_model,Da_dl,B,criterion)
        subpolicies=get_topn_subpolicies(subpolicies,N)
        transform.extend([subpolicy[0] for subpolicy in subpolicies])
    return transform
from tqdm import tqdm

def fast_auto_augment(model,Da_dl,B=300,T=2,N=10):
    transform_list=[]
    transform_candidates = DEFALUT_CANDIDATES
   
#     for i,Da_dl in enumerate(Da_dls):
#     model=models[i]
    transform=process_fn(model,Da_dl,T,transform_candidates,B,N)
           
    transform_list.extend(transform)
            
#     transform_list=transforms.RandomChoice(transform_list)
    return transform_list
import pickle
def main():
    k=0
#     Da_dls=[]
#     models=[]
#     for k in range(5):
#         with open('/kaggle/input/linshi/valid_dl{}.txt'.format(k),'rb') as f:
#             valid_dl=pickle.load(f)
#             ds=valid_dl.dataset
#             new_valid_dl=Data.DataLoader(ds,batch_size=32,shuffle=True)
#             Da_dls.append(new_valid_dl)
#         with open('/kaggle/input/linshi/model{}.txt'.format(k),'rb') as f:
#             model=pickle.load(f)
#             models.append(model)
    with open('/kaggle/input/linshi/valid_dl{}.txt'.format(k),'rb') as f:
            valid_dl=pickle.load(f)
            ds=valid_dl.dataset
            new_valid_dl=Data.DataLoader(ds,batch_size=32,shuffle=True)
    with open('/kaggle/input/linshi/model{}.txt'.format(k),'rb') as f:
       model=pickle.load(f)        
    transform_list=fast_auto_augment(model,new_valid_dl)
    print(transform_list)
    file=open('transform_list{}.txt'.format(k),'wb+')
    pickle.dump(transform_list,file)
    file.close()
main()