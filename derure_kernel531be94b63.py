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
!pip install geffnet
!pip install git+https://github.com/pabloppp/pytorch-tools -U
import torch
from fastai.vision import *
import torch.utils.data as Data
import cv2
import albumentations as A
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import geffnet
import torch.nn.functional as F
from torchtools.optim import RangerLars
from collections import defaultdict
import torchvision.transforms as transforms
from PIL import Image as I
import copy
import random
train_df=pd.read_csv('/kaggle/input/futurefish/training.csv')
label_csv=pd.read_csv('/kaggle/input/futurefish/species.csv')
test_csv=pd.read_csv('/kaggle/input/futurefish/annotation.csv')

image_dir='/kaggle/input/futurefish/data/data'
train_names=[]
image_labels={}
valid_names=[]
label_map_image=defaultdict(list)
label_set=set()
for i in range(train_df.shape[0]):
    info=train_df.iloc[i]
    name=info['FileID']
    name=os.path.join(image_dir,name+'.jpg')
    train_names.append(name)
    label=info['SpeciesID']
    image_labels[name]=label
    label_set.add(label)
    label_map_image[label].append(name)
for i in range(test_csv.shape[0]):
    info=test_csv.iloc[i]
    name=info['FileID']
    name=os.path.join(image_dir,name+'.jpg')
    valid_names.append(name)
    label=info['SpeciesID']
    image_labels[name]=label
    label_map_image[label].append(name)
label_set=list(label_set)
print(len(label_set))

transform_train=transforms.Compose([
        transforms.Resize([448,448]),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])      ]

)
transform_valid=transforms.Compose([
    transforms.Resize([448,448]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_label_map_image=defaultdict(list)
for name in train_names:
    label=image_labels[name]
    train_label_map_image[label].append(name)

    
batch_names=copy.deepcopy(train_names)
batch_label_map_image=copy.deepcopy(train_label_map_image)
batch_index=[]
while True:
    name = batch_names[0]
    batch = []
    batch.append(train_names.index(name))
    batch_names.remove(name)
    label = image_labels[name]
  
    batch_label = list(set(label_set) ^ set([label]))
    random.shuffle(batch_label)

    for i in range(11):
        label = batch_label[i]
        label_names=batch_label_map_image[label]
        if len(label_names)==0:
            label_names=train_label_map_image[label]
            name=np.random.choice(label_names)
        else:
            name=np.random.choice(label_names)
            batch_label_map_image[label].remove(name)
        if name in batch_names:
            batch_names.remove(name)
        index = train_names.index(name)
        batch.append(index)

    batch_index.append(batch)
    if len(batch_names) == 0:
        break
print("origin_size:{},current_size:{}".format(len(train_names)/16,len(batch_index)))
print(len(batch_index[-1]))

class  myDataset(Data.Dataset):
    def __init__(self, names, image_labels, transform):
        super(myDataset, self).__init__()
        self.names = names
        self.image_labels = image_labels
        self.transform = transform

    def __getitem__(self, index):
        name = self.names[index]
        if type(name) == list:
            name = name[0]
        label = self.image_labels[name]
        image = I.open(name)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.names)


from torch.nn.parameter import Parameter


class CBatchNorm2d(nn.Module):
    def __init__(self, num_features, myeps=1e-5, mymomentum=0.1, affine=True,
                 track_running_stats=True,
                 buffer_num=0, rho=1.0,
                 burnin=0, two_stage=True,
                 FROZEN=False, out_p=False,**kwargs):
        super(CBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = myeps
        self.momentum = mymomentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.buffer_num = buffer_num
        self.max_buffer_num = buffer_num
        self.rho = rho
        self.burnin = burnin
        self.two_stage = two_stage
        self.FROZEN = FROZEN
        self.out_p = out_p

        self.iter_count = 0
        self.pre_mu = []
        self.pre_meanx2 = []  # mean(x^2)
        self.pre_dmudw = []
        self.pre_dmeanx2dw = []
        self.pre_weight = []
        self.ones = torch.ones(self.num_features).cuda()

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def _update_buffer_num(self):
        if self.two_stage:
            if self.iter_count > self.burnin:
                self.buffer_num = self.max_buffer_num
            else:
                self.buffer_num = 0
        else:
            self.buffer_num = int(self.max_buffer_num * min(self.iter_count / self.burnin, 1.0))

    def forward(self, input, weight):
        # deal with wight and grad of self.pre_dxdw!
        self._check_input_dim(input)
        y = input.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(input.size(1), -1)

        # burnin
        if self.training and self.burnin > 0:
            self.iter_count += 1
            self._update_buffer_num()

        if self.buffer_num > 0 and self.training and input.requires_grad:  # some layers are frozen!
            # cal current batch mu and sigma
            cur_mu = y.mean(dim=1)
            cur_meanx2 = torch.pow(y, 2).mean(dim=1)
            cur_sigma2 = y.var(dim=1)
            # cal dmu/dw dsigma2/dw
            dmudw = torch.autograd.grad(cur_mu, weight, self.ones, retain_graph=True)[0]
            dmeanx2dw = torch.autograd.grad(cur_meanx2, weight, self.ones, retain_graph=True)[0]
            # update cur_mu and cur_sigma2 with pres
            mu_all = torch.stack(
                [cur_mu, ] + [tmp_mu + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for
                              tmp_mu, tmp_d, tmp_w in zip(self.pre_mu, self.pre_dmudw, self.pre_weight)])
            meanx2_all = torch.stack(
                [cur_meanx2, ] + [tmp_meanx2 + (self.rho * tmp_d * (weight.data - tmp_w)).sum(1).sum(1).sum(1) for
                                  tmp_meanx2, tmp_d, tmp_w in
                                  zip(self.pre_meanx2, self.pre_dmeanx2dw, self.pre_weight)])
            sigma2_all = meanx2_all - torch.pow(mu_all, 2)

            # with considering count
            re_mu_all = mu_all.clone()
            re_meanx2_all = meanx2_all.clone()
            re_mu_all[sigma2_all < 0] = 0
            re_meanx2_all[sigma2_all < 0] = 0
            count = (sigma2_all >= 0).sum(dim=0).float()
            mu = re_mu_all.sum(dim=0) / count
            sigma2 = re_meanx2_all.sum(dim=0) / count - torch.pow(mu, 2)

            self.pre_mu = [cur_mu.detach(), ] + self.pre_mu[:(self.buffer_num - 1)]
            self.pre_meanx2 = [cur_meanx2.detach(), ] + self.pre_meanx2[:(self.buffer_num - 1)]
            self.pre_dmudw = [dmudw.detach(), ] + self.pre_dmudw[:(self.buffer_num - 1)]
            self.pre_dmeanx2dw = [dmeanx2dw.detach(), ] + self.pre_dmeanx2dw[:(self.buffer_num - 1)]

            tmp_weight = torch.zeros_like(weight.data)
            tmp_weight.copy_(weight.data)
            self.pre_weight = [tmp_weight.detach(), ] + self.pre_weight[:(self.buffer_num - 1)]

        else:
            x = y
            mu = x.mean(dim=1)
            cur_mu = mu
            sigma2 = x.var(dim=1)
            cur_sigma2 = sigma2

        if not self.training or self.FROZEN:
            y = y - self.running_mean.view(-1, 1)
            # TODO: outside **0.5?
            if self.out_p:
                y = y / (self.running_var.view(-1, 1) + self.eps) ** .5
            else:
                y = y / (self.running_var.view(-1, 1) ** .5 + self.eps)

        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * cur_mu
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * cur_sigma2
            y = y - mu.view(-1, 1)
            # TODO: outside **0.5?
            if self.out_p:
                y = y / (sigma2.view(-1, 1) + self.eps) ** .5
            else:
                y = y / (sigma2.view(-1, 1) ** .5 + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'buffer={max_buffer_num}, burnin={burnin}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class Residual(nn.Module):
    def __init__(self, in_channel, R=8, k=2):
        super(Residual, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ELU(inplace=True)
        self.R = R
        self.k = k
        out_channel = int(in_channel / R)
        self.fc1 = nn.Linear(in_channel, out_channel)
        fc_list = []
        for i in range(k):
            fc_list.append(nn.Linear(out_channel, 2 * in_channel))
        self.fc2 = nn.ModuleList(fc_list)

    def forward(self, x):
        x = self.avg(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.relu(x)
        result_list = []
        for i in range(self.k):
            result = self.fc2[i](x)
            result = 2 * torch.sigmoid(result) - 1
            result_list.append(result)
        return result_list


class Dynamic_relu_b(nn.Module):
    def __init__(self, inchannel, R=8, k=2):
        super(Dynamic_relu_b, self).__init__()
        self.lambda_alpha = 1
        self.lambda_beta = 0.5
        self.R = R
        self.k = k
        self.init_alpha = torch.zeros(self.k)
        self.init_beta = torch.zeros(self.k)
        self.init_alpha[0] = 1
        self.init_beta[0] = 1
        for i in range(1, k):
            self.init_alpha[i] = 0
            self.init_beta[i] = 0

        self.residual = Residual(inchannel)

    def forward(self, input):
        delta = self.residual(input)
        in_channel = input.shape[1]
        bs = input.shape[0]
        alpha = torch.zeros((self.k, bs, in_channel),device=input.device)
        beta = torch.zeros((self.k, bs, in_channel),device=input.device)
        for i in range(self.k):
            for j, c in enumerate(range(0, in_channel * 2, 2)):
                alpha[i, :, j] = delta[i][:, c]
                beta[i, :, j] = delta[i][:, c + 1]
        alpha1 = alpha[0]
        beta1 = beta[0]
        max_result = self.dynamic_function(alpha1, beta1, input, 0)
        for i in range(1, self.k):
            alphai = alpha[i]
            betai = beta[i]
            result = self.dynamic_function(alphai, betai, input, i)
            max_result = torch.max(max_result, result)
        return max_result
    def dynamic_function(self, alpha, beta, x, k):
        init_alpha = self.init_alpha[k]
        init_alpha=init_alpha.to(x.device)
        init_beta = self.init_beta[k]
        init_beta=init_beta.to(x.device)
        # lambda_alpha=self.lambda_alpha.to(x.device)
        # lambda_beta=self.lambda_beta.to(x.device)
        alpha = init_alpha +  self.lambda_alpha* alpha
        beta = init_beta + self.lambda_beta * beta
        bs = x.shape[0]
        channel = x.shape[1]
        results = torch.zeros_like(x,device=x.device)
        results = x * alpha.view(bs, channel, 1, 1) + beta.view(bs, channel, 1, 1)
        return results


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        backbone = geffnet.efficientnet_b3(pretrained=True)
#         state_dict=torch.load('/kaggle/input/gettnet-0-9-7/efficientnet_b3_ra-a5e2fbc7.pth')

#         model_dict=backbone.state_dict()
#         # 1. filter out unnecessary keys
#         pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
#         # 2. overwrite entries in the existing state dict
#         model_dict.update(pretrained_dict)
#         backbone.load_state_dict(model_dict)
#         print(backbone)
        act1 = Dynamic_relu_b(40)
        act2 = Dynamic_relu_b(1536)
        self.backbone = torch.nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            act1,
            backbone.blocks,
            backbone.conv_head,
            backbone.bn2,
            act2
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


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        '''

        :param x: [batch_size,feat_dim]
        :param labels: [batch_size]
        :return:
        '''
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        classes = classes
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss



import torch.nn.functional as Fu
class Criterion:
    def __init__(self):
        self.triplet_criterion = TripletLoss()
        self.cls_criterion = nn.BCEWithLogitsLoss()


    def __call__(self, global_feat, local_feat, cls_score, label):
        global_loss = self.triplet_criterion('global', global_feat, label)

        local_loss = self.triplet_criterion('local', local_feat, label)
        label = one_hot_smooth_label(label, 20)
        cls_loss = self.cls_criterion(cls_score, label)
        l_bcn=self.get_bcn(cls_score)
      
        return global_loss+local_loss+cls_loss+l_bcn

    def get_bcn(self,cls_score):
        p=F.softmax(cls_score,dim=1)
        L_bcn=torch.norm(p.t().mm(p),p='nuc')
        return L_bcn




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path,patience=4, best_score=None, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path=checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_metric, model):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        state = {'best_metric': metric, 'state': model.state_dict()}
        torch.save(state, self.checkpoint_path)


from fastai.vision import *
from fastai import *


def evaluate(model,valid_dl):
    steps=len(valid_dl)
    device=torch.device('cuda:0')

    step_metric=0
    with torch.no_grad():
        for images,labels in valid_dl:
            images=images.to(device).float()
            global_feat,local_feat,cls_score=model(images)
            cls_score=cls_score.to('cpu')
            metric=accuracy(cls_score,labels)
            step_metric+=metric
    metric=step_metric/steps

    print('metric:{}'.format(metric))
    return metric

from tqdm import tqdm
import pickle


class flat_and_anneal(nn.Module):
    def __init__(self, epochs, anneal_start=0.5, base_lr=0.001, min_lr=0):
        super(flat_and_anneal, self).__init__()
        self.epochs = epochs
        self.anneal_start = anneal_start
        self.base_lr = base_lr
        self.min_lr = min_lr

    def forward(self, epoch, optimizer):
        if epoch >= 15:
            epoch = epoch - 15
            for param in optimizer.param_groups:
                lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / 5)) / 2
                param['lr'] = lr


def main(train_dl, valid_dl, k):
    criterion = Criterion()
    checkpoint_path = 'test1_step{}.pt'.format(k)
    early_stop = EarlyStopping(checkpoint_path)

    model = myNet()
    device = torch.device('cuda:0')
    model = model.to(device)

    epochs = 50
    optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=0.00001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)

    for epoch in range(epochs):
        with tqdm(total=len(train_dl)) as pbar:
                train_loss = 0
                steps = len(train_dl)
           
                for image, labels in train_dl:
                    model.train()
                    optimizer.zero_grad()

                    image = image.to(device).float()
                    global_feat, local_feat, cls_score= model(image)
                    global_feat = global_feat.to('cpu')
                    cls_score = cls_score.to('cpu')
                    local_feat = local_feat.to('cpu')

                    loss = criterion(global_feat, local_feat, cls_score, labels)
                    train_loss += loss
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)
                print('epoch:{},train_loss:{}'.format(epoch, train_loss / steps))
                model.eval()
                metric = evaluate(model, valid_dl)
                early_stop(metric, model)
                scheduler.step()


class myBatchSampler(Data.Sampler):
    def __init__(self, batch_index, batch_size, drop_last):

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.batch_index = batch_index

    def __iter__(self):
        batch = []
        for batch_index in self.batch_index:
            for index in batch_index:
                batch.append(index)

            yield batch
            batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.batch_index)

train_ds=myDataset(train_names,image_labels,transform_train)
valid_ds=myDataset(valid_names,image_labels,transform_valid)
sampler=myBatchSampler(batch_index,batch_size=16,drop_last=True)
train_dl=Data.DataLoader(train_ds,batch_sampler=sampler)
valid_dl=Data.DataLoader(valid_ds,batch_size=8,drop_last=True)
k=0
main(train_dl,valid_dl,k)


