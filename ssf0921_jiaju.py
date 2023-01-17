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
!pip install git+https://github.com/qubvel/segmentation_models.pytorch
!pip install git+https://github.com/pabloppp/pytorch-tools -U
    
import torch
import torch.utils.data as Data
from fastai.vision import *
import cv2
import albumentations as A
from torchtools.optim import RangerLars
import math
from collections import defaultdict
import segmentation_models_pytorch as smp
torch.backends.cudnn.benchmark=True
import torchvision.transforms as transforms
from collections import defaultdict
from PIL import Image as I
import random
train_infos=[]
test_infos=[]
train_path=['/kaggle/input/cls-jiaju/cls_data/train_data/sofa','/kaggle/input/cls-jiaju/cls_data/train_data/bed']
test_path=['/kaggle/input/cls-jiaju/cls_data/val_data/sofa/images','/kaggle/input/cls-jiaju/cls_data/val_data/bed/images']
label_map_images=defaultdict(list)
label_dict={'sofa':0,'bed':1}
for i in range(2):
    path=train_path[i]
    names=get_image_files(path)
    for name in names:
        info={}
        name=str(name)
        image=I.open(name)
        label=str(path).split('/')[-1]
        label=label_dict[label]
        info['image']=image
        info['label']=label
        label_map_images[label].append(name)
        train_infos.append(info)

for i in range(2):
    path = test_path[i]
    names = get_image_files(path)
    for name in names:
        info = {}
        name = str(name)
        image = I.open(name)
        label = str(path).split('/')[-2]
        label = label_dict[label]
        info['image'] = image
        info['label'] = label
        test_infos.append(info)

class myDataset(Data.Dataset):
    def __init__(self,info,transofrm,label_map_image=None):
        super(myDataset, self).__init__()
        self.info=info
        self.transform=transofrm
        self.label_map_image=label_map_image

    def aug_image(self,image):
        image=image.convert("RGB")
        image=self.transform(image)
        return image

    def __getitem__(self, index):
        info=self.info[index]

        image=self.aug_image(info['image'])
        label=info['label']

       
        negative_label=1-label
        negative_path=random.choice(self.label_map_image[negative_label])
        
        
        negative_image=self.aug_image(I.open(negative_path))
        
        return [image,negative_image],[label,negative_label]
       



    def __len__(self):
        return len(self.info)

class TestDataset(Data.Dataset):
    def __init__(self,info,transofrm,label_map_image=None):
        super(TestDataset, self).__init__()
        self.info=info
        self.transform=transofrm
        self.label_map_image=label_map_image

    def aug_image(self,image):
        image=self.transform(image)
        return image
        
    def __getitem__(self, index):
        info=self.info[index]

        image=self.aug_image(info['image'])
        label=info['label']

       
   

        return image,label
       



    def __len__(self):
        return len(self.info)


def train_collate(batch):
        batch_size = len(batch)
        images = []
        labels = []
        for b in range(batch_size):
            if batch[b][0] is None:
                continue
            else:
                images.extend(batch[b][0])
                labels.extend(batch[b][1])
        images = torch.stack(images, 0)
        labels = torch.from_numpy(np.array(labels))
        return images, labels

def create_dataloader(infos,transform,batch_size,label_map_image=None,shuffle=False,collate_fn=None):
    if label_map_image:
        dataset=myDataset(infos,transform,label_map_image)
    else:
        dataset=TestDataset(infos,transform)
    dataloader=Data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn)
    return dataloader



class Fencemask:
    def __init__(self, image_size,gmin=20, gmax=30, wmin=40, wmax=60, rotate=45):
        self.gmin = gmin
        self.gmax = gmax
        self.wmin = wmin
        self.wmax = wmax
        self.rotate = rotate
        self.image_size = image_size

    def __call__(self, image, p):

        bs = image.shape[0]
        masks = torch.zeros_like(image, device=image.device)
        prob = np.random.rand(1)
        if prob > p:
            for j in range(bs):
                gx = np.random.randint(self.gmin, self.gmax)
                gy = np.random.randint(self.gmin, self.gmax)
                wx = np.random.randint(self.wmin, self.wmax)
                wy = np.random.randint(self.wmin, self.wmax)

                x_step = int(self.image_size / (gx + wx))
                y_step = int(self.image_size / (gy + wy))

                x_width = gx + wx
                y_width = gy + wy

                mask = np.ones((self.image_size, self.image_size, 3))

                for i in range(x_step):
                    start = i * x_width + 20

                    end = start + gx
                    mask[start:end, :, :] = 0

                for i in range(y_step):
                    start = i * x_width + 20

                    end = start + gy
                    mask[:, start:end, :] = 0

                mask = A.Rotate(self.rotate)(image=mask)['image']
                mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).to(image.device)
                masks[j] = mask

            image = image * masks
        else:
            image = image

        return image


class attention(nn.Module):
    def __init__(self, feature_size=512, add_scaling_factor=False, return_attention_weights=False):
        super(attention, self).__init__()
        self.att_len = nn.Linear(feature_size, 1)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.add_scaling_factor = add_scaling_factor
        self.return_attention_weights = return_attention_weights
        self.atten = []

    def forward(self, patches):
        # patch-> (N, num_patches, feature_size)
        atten_weights = self.att_len(patches).squeeze(2)  # (N,num_patches)
        softmax_weights = self.softmax(atten_weights)  # (N,num_patches)
        if (self.add_scaling_factor):
            softmax_weights /= np.sqrt(512)
            # self.atten = np.append(self.atten, torch.argmax(softmax_weights, 1).cpu().data.numpy())
        patch_attention_encoding = (patches * softmax_weights.unsqueeze(2)).sum(1)  # (N, feature_size)
        if (self.return_attention_weights):
            return patch_attention_encoding, softmax_weights
        return patch_attention_encoding

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output
class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        backbone=smp.encoders.get_encoder('efficientnet-b3',weights='imagenet')
        self.backbone=backbone
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.embedding_fc = nn.Linear(384, 1024)
        self.fc = nn.Linear(1024, 2)
        self.local_conv = nn.Conv2d(384, 512, 1)
        self.local_bn = nn.BatchNorm2d(512)
        self.local_bn.bias.requires_grad = False
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.constant_(self.fc.bias, 0)
        nn.init.kaiming_normal_(self.embedding_fc.weight, mode='fan_out')
        nn.init.constant_(self.embedding_fc.bias, 0)
        self.mask=Fencemask(640)

    def forward(self,x):
       

        x=self.backbone(x)[-1]
        local_feat=x
        x=self.avg_pool(x)
        x=x.view(x.shape[0],-1)

        embedding=self.embedding_fc(x)
        embedding=nn.functional.normalize(embedding,p=2,dim=1)

        out=self.fc(embedding)



        local_feat = torch.mean(local_feat, -1, keepdim=True)
        local_feat = self.local_bn(self.local_conv(local_feat))
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        local_feat = l2_norm(local_feat, axis=-1)
        return out,embedding,local_feat



def one_hot_smooth_label(x, num_class, smooth=0.1):
    num = x.shape[0]
    labels = torch.zeros((num, num_class))
    for i in range(num):
        labels[i][x[i]] = 1
    labels = (1 - (num_class - 1) / num_class * smooth) * labels + smooth / num_class
    return labels


import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2
        self.scale_neg = 50
        self.count = 0

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):

            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < (1 - epsilon)]
            if len(pos_pair_) == 0:
                print('MS False  count:{}'.format(self.count))
                self.count += 1
                continue

            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss

class Criterion:
    def __init__(self):
        self.cls_criterion=nn.BCEWithLogitsLoss()
        self.ms_criterion=MultiSimilarityLoss()
        self.local_criterion=TripletLoss()

    def __call__(self,logits,embedding,local_feat,label):
        global_loss = self.ms_criterion(embedding, label)
        local_loss = self.local_criterion('local', local_feat, label)
        label = one_hot_smooth_label(label, 2)
        cls_loss = self.cls_criterion(logits, label)

        return global_loss + local_loss + cls_loss


    """Early stops the training if validation loss doesn't improve after a given patience."""
class EarlyStopping:
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
    step_loss=0
    step_metric=0
    with torch.no_grad():
        for images,labels in valid_dl:
            images=images.to(device).float()
            cls_score,_,_=model(images)
            cls_score=cls_score.to('cpu')
            metric=accuracy(cls_score,labels)
            step_metric+=metric
    metric=step_metric/steps
    valid_loss=step_loss/steps
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
        if epoch >= 60:
            epoch = epoch - 60
            for param in optimizer.param_groups:
                lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / 5)) / 2
                param['lr'] = lr


def main(train_dl, valid_dl):
    criterion = Criterion()
    checkpoint_path = 'base_step.pt'
    early_stop = EarlyStopping(checkpoint_path)

    model = myNet()
    device = torch.device('cuda:0')
    model = model.to(device)

    epochs = 100
    optimizer = RangerLars(model.parameters(), lr=0.001)
    scheduler = flat_and_anneal(epochs)
    init_p = 0.8
    for epoch in range(epochs):
        with tqdm(total=len(train_dl)) as pbar:
            train_loss = 0
            steps = len(train_dl)
          

            for image, labels in train_dl:
                model.train()
                optimizer.zero_grad()

                image = image.to(device).float()
                logits,embedding,local_feat = model(image)
                logits=logits.to('cpu')
                local_feat=local_feat.to('cpu')
                embedding = embedding.to('cpu')
                loss = criterion( logits,embedding,local_feat,labels)
                train_loss += loss
                loss.backward()
                optimizer.step()
                pbar.update(1)
            print('epoch:{},train_loss:{}'.format(epoch, train_loss / steps))
            model.eval()
            metric = evaluate(model, valid_dl)
            early_stop(metric, model)
            scheduler(epoch, optimizer)






import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)





def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))





        
        
        
        
class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.p=0.5
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        p=np.random.rand(1)
        if p>self.p:
            for op, minval, maxval in ops:
                val = (float(self.m) / 30) * float(maxval - minval) + minval
                img = op(img, val)
        else:
            img=img
           

        return img
rand_augment=RandAugment(3,9)

transform_train=transforms.Compose([
        transforms.Resize([640,640]),

        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomRotation(45),
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing()

]

)
transform_valid=transforms.Compose([
    transforms.Resize([640,640]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader=create_dataloader(train_infos,transform_train,batch_size=4,label_map_image=label_map_images,shuffle=True,collate_fn=train_collate)
valid_loader=create_dataloader(test_infos,transform_valid,batch_size=8)
main(train_loader,valid_loader)
import matplotlib.pyplot as plt
%matplotlib inline
image=I.open('/kaggle/input/cls-jiaju/cls_data/train_data/sofa/Screen-Shot-2018-01-12-at-10.26.16-696x385.png')
plt.imshow(image)
image=np.array(image)
image.shape
