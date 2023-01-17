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
from collections import defaultdict
from torchtools.optim import RangerLars
train_df=pd.read_csv('/kaggle/input/futurefish/training.csv')
train_df.head(20)
label_csv=pd.read_csv('/kaggle/input/futurefish/species.csv')
label_csv.head()
test_csv=pd.read_csv('/kaggle/input/futurefish/annotation.csv')
test_csv.head(20)
import torchvision.transforms as transforms
from PIL import Image as I
class TrainDataset(Data.Dataset):
    def __init__(self,names,image_labels,labels_map_images,transform):
        super(TrainDataset,self).__init__()
        self.names=names
        self.image_labels=image_labels
        self.transform=transform
    
        self.num_class=20
        self.labels_map_images=labels_map_images
        
    def read_image(self,name):
        image=I.open(name)
        image=self.transform(image)
        return image
        
    def __getitem__(self,index):
        
        name=self.names[index]
       
        if type(name)==list:
            
            name=name[0]
        label=self.image_labels[name]
        negative_label=np.random.choice(list(set(list(range(self.num_class)))^set([label])))
        
        negative_name=np.random.choice(self.labels_map_images[negative_label])
        positive_name=np.random.choice(list(set(self.labels_map_images[label])^set([name])))
        
        
        image=self.read_image(name)
        
        positive_image=self.read_image(positive_name)
        
        negative_image=self.read_image(negative_name)
        return [image,positive_image,negative_image],[label,label,negative_label]
        
    def __len__(self):
        return len(self.names)
    
class TestDataset(Data.Dataset):
    def __init__(self,names,image_labels,transform):
        super(TestDataset,self).__init__()
        self.names=names
        self.image_labels=image_labels
        self.transform=transform
        
    def __getitem__(self,index):
        name=self.names[index]
        if type(name)==list:
            name=name[0]
        label=self.image_labels[name]
        image=I.open(name)
        image=self.transform(image)
        return image,label
        
    def __len__(self):
        return len(self.names)

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
class Residual(nn.Module):
    def __init__(self, in_channel, R=8, k=2):
        super(Residual, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
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
    
class k_max_pool(nn.Module):
    def __init__(self, k=4):
        super(k_max_pool, self).__init__()
        self.k = k
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.topk(self.k, dim=-1).values
        x = self.pool(x).squeeze()
        return x


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class mySwish(nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)


import torchvision


class Location_Model(nn.Module):
    def __init__(self):
        super(Location_Model, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        act1 = Dynamic_relu_b(64)
        act2 = Dynamic_relu_b(256)
        pool = k_max_pool()
        swish = mySwish()
        block_0 = model.layer1[0]
        block_0.relu = swish
        block_1 = model.layer1[1]
        block_1.relu = swish
        block_2 = model.layer1[2]
        block_2.relu = swish
        self.model = nn.Sequential(
            model.conv1,
            model.bn1,
            act1,
            model.maxpool,
            block_0,
            block_1,
            block_2
        )

    def forward(self, x):
        x = F.interpolate(x, size=(40, 40), mode='bilinear')

        x = self.model(x)
        x = torch.mean(x, dim=1).unsqueeze(1)
        return x


import torch
import torch.nn as nn
import numpy as np


class MaxLayer(nn.Module):
    def __init__(self):
        super(MaxLayer, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x_max = torch.max(x, dim=1).values
        return x_max


class MinLayer(nn.Module):
    def __init__(self):
        super(MinLayer, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x_min = torch.min(x, dim=1).values
        return x_min


class TrainSubtractionLayer(nn.Module):
    def __init__(self, sub=0.3):
        super(TrainSubtractionLayer, self).__init__()
        sub = np.array([sub])
        sub = torch.from_numpy(sub)[0].float()
        self.sub = nn.Parameter(sub, requires_grad=True)

    def forward(self, x):
        return x - self.sub


class DivisionLayer(nn.Module):
    def __init__(self):
        super(DivisionLayer, self).__init__()

    def forward(self, x, divisor):
        return x / divisor


class Min_Max_Normalize(nn.Module):
    def __init__(self):
        super(Min_Max_Normalize, self).__init__()
        self.min_layer = MinLayer()
        self.max_layer = MaxLayer()
        self.divisor = DivisionLayer()

    def forward(self, x):
        x_max = self.max_layer(x).view(x.shape[0], 1, 1, 1)
        x_min = self.min_layer(x).view(x.shape[0], 1, 1, 1)

        x = self.divisor(x - x_min, x_max - x_min)
        return x


class ThresholdLayer(nn.Module):
    def __init__(self):
        super(ThresholdLayer, self).__init__()
        self.sub_layer = TrainSubtractionLayer()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.sub_layer(x))


class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.relu = mySwish()
        self.fc2 = nn.Linear(128, 3)
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0], dtype=torch.float))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Affnet(nn.Module):
    def __init__(self):
        super(Affnet, self).__init__()
        min_max_normalize = Min_Max_Normalize()
        threshold_layer = ThresholdLayer()
        self.preprocess_model = nn.Sequential(
            min_max_normalize,
            threshold_layer
        )
        self.pool1 = nn.MaxPool2d((1, 10))
        self.pool2 = nn.MaxPool2d((10, 1))
        self.sub1 = SubModel()
        self.sub2 = SubModel()

    def forward(self, x):
        x = self.preprocess_model(x)

        xv = self.pool2(x).squeeze()
        xv = self.sub2(xv)
        xh = self.pool1(x).squeeze()
        xh = self.sub1(xh)
        theta = torch.zeros((x.shape[0], 6), device=x.device)
        theta[:, 0] = xh[:, 0]
        theta[:, 3] = xv[:, 0]
        theta[:, 1] = xh[:, 1]
        theta[:, 4] = xv[:, 1]
        theta[:, 2] = xh[:, 2]
        theta[:, 5] = xh[:, 2]
        return theta

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Fencemask:
    def __init__(self,gmin=10,gmax=25,wmin=40,wmax=60,rotate=30):
        self.gmin=gmin
        self.gmax=gmax
        self.wmin=wmin
        self.wmax=wmax
        self.rotate=rotate
        self.image_size=448
        
    def __call__(self,image):
        
        bs=image.shape[0]
        masks=torch.zeros_like(image,device=image.device)
        for j in range(bs):
            gx=np.random.randint(self.gmin,self.gmax)
            gy=np.random.randint(self.gmin,self.gmax)
            wx=np.random.randint(self.wmin,self.wmax)
            wy=np.random.randint(self.wmin,self.wmax)
            
            x_step=int(self.image_size/(gx+wx))
            y_step=int(self.image_size/(gy+wy))

            x_width=gx+wx
            y_width=gy+wy

            mask=np.ones((self.image_size,self.image_size,3))

            for i in range(x_step):

                  start=i*x_width+20

                  end=start+gx
                  mask[start:end,:,:]=0

            for i in range(y_step):

                  start=i*x_width+20

                  end=start+gy
                  mask[:,start:end,:]=0


            mask=A.Rotate(self.rotate)(image=mask)['image']
            mask=torch.from_numpy(np.transpose(mask,(2,0,1))).to(image.device)
            masks[j]=mask
            
            
        image=image*masks
   
        return image
class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        backbone = geffnet.efficientnet_b3(pretrained=True)
        act1 = Dynamic_relu_b(40)
        act2 = Dynamic_relu_b(1536)
        self.AttNet = Location_Model()
        self.AffNet = Affnet()
        self.backbone = torch.nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            act1,
            backbone.blocks,
            backbone.conv_head,

            backbone.bn2,
            act2,
#             nn.Conv2d(1536,60,1)
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
        
        self.mask=Fencemask()
  

    def forward(self, x):
        image = x
        x = self.AttNet(x)
        Att_M = x

        x = self.AffNet(x)
        Aff_theta = x
        theta = torch.zeros((x.shape[0], 2, 3), device=x.device)
        theta[:, 0, :] = x[:, :3]
        theta[:, 1, :] = x[:, 3:]
        grid = F.affine_grid(theta, image.size())
        x = F.grid_sample(image, grid)

        del image
        del theta
#         with torch.no_grad():
#             x=self.mask(x)
       
        x = self.backbone(x)
        
      
        
        M = torch.mean(x, dim=1).unsqueeze(1)


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
        return global_feat,local_feat,out, Att_M, Aff_theta, M


from skimage import measure



def one_hot_smooth_label(x, num_class, smooth=0.1):
    num = x.shape[0]
    labels = torch.zeros((num, 20))
    for i in range(num):
        labels[i][x[i]] = 1
    labels = (1 - (num_class - 1) / num_class * smooth) * labels + smooth / num_class
    return labels


import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    
class Criterion:
    def __init__(self):
        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.triplet_criterion = TripletLoss()
        self.num_class=20

        self.center_criterion = CenterLoss(20, 1536)

    def normalize_transforms(self, transforms, W, H):
        transforms[0, 0] = transforms[0, 0]
        transforms[0, 1] = transforms[0, 1] * H / W
        transforms[0, 2] = transforms[0, 2] * 2 / W + transforms[0, 0] + transforms[0, 1] - 1

        transforms[1, 0] = transforms[1, 0] * W / H
        transforms[1, 1] = transforms[1, 1]
        transforms[1, 2] = transforms[1, 2] * 2 / H + transforms[1, 0] + transforms[1, 1] - 1

        return transforms

    def get_affine(self, M):
        M = (M > 0).float()
        parameters = torch.zeros((M.shape[0], 6))
        for i, m in enumerate(M):
            m = m.squeeze().numpy()
            component = measure.label(m, connectivity=2)
            prob = measure.regionprops(component)
            if len(prob)==0:
                miny=0
                minx=0
                maxy=10
                maxx=10
            else:
                miny, minx, maxy, maxx = prob[0].bbox
            miny = (miny) * 30 - 1
            minx = (minx) * 30 - 1
            maxx = (maxx) * 30 - 1
            maxy = (maxy) * 30 - 1
            point1 = np.float32([[minx, miny], [maxx, maxy], [minx, maxy]])
            point2 = np.float32([[0, 0], [300, 300], [0, 300]])

            M = cv2.getAffineTransform(point1, point2)
            a = torch.ones((3, 3))
            a[0][0] = M[0][0]
            a[1][0] = M[0][1]
            a[2][0] = M[0][2]
            a[0][1] = M[1][0]
            a[1][1] = M[1][1]
            a[2][1] = M[1][2]
            a[0][2] = 0
            a[1][2] = 0
            a[2][2] = 1
            a = a.inverse()
            m = np.array([
                [a[0][0], a[1][0], a[2][0]],
                [a[0][1], a[1][1], a[2][1]]
            ])
            m = self.normalize_transforms(m, 300, 300)
            theta = torch.from_numpy(m).float()
            parameters[i] = theta.view(-1, 6)[0]
        return parameters
    
    
    

    def get_L_dis(self,global_feat,label):
        
        bs=global_feat.shape[0]
        channel=global_feat.shape[1]
        theta=int(channel/self.num_class)
        F_all=torch.zeros((bs,self.num_class),device=global_feat.device)
        for i in range(self.num_class):
            F_i=global_feat[:,int(i*theta):int((i+1)*theta)]
       
            F_i=F_i.view(F_i.shape[0],F_i.shape[1],-1)
            Mi=torch.zeros((bs,theta),device=global_feat.device)
            for j in range(bs):
                index=np.random.choice(range(theta),int(theta/2))
                Mi[j,index]=1
            F_i=F_i*(Mi.unsqueeze(-1))
           
            F_i=torch.max(F_i,1).values
         
            F_i=nn.AdaptiveAvgPool1d(1)(F_i.unsqueeze(1)).squeeze()
           
            F_all[:,i]=F_i
        F_all=F.softmax(F_all,dim=1)
       
        L_bis=nn.BCEWithLogitsLoss()(F_all,label)
        return L_bis


    def get_L_div(self,global_feat):
        bs = global_feat.shape[0]
        channel = global_feat.shape[1]
        theta =int( channel / self.num_class)
        F_all = torch.zeros((bs,self.num_class), device=global_feat.device)
        for i in range(self.num_class):
            F_i = global_feat[:,int( i * theta):int((i + 1) * theta)]
            F_i = F_i.view(F_i.shape[0], F_i.shape[1], -1)
       
            F_i=F.softmax(F_i,dim=-1)
       
            F_i = torch.max(F_i,1).values
        
            F_i= F_i.sum(dim=1)
      
            F_all[:,i]=F_i
        L_div=F_all.sum(dim=1)/self.num_class
        L_div=L_div.sum()/bs
        
        return L_div
    

    def __call__(self,  global_feat, local_feat,cls_score, label, Att_M, Aff_theta, M):
        global_loss = self.triplet_criterion('global', global_feat, label)
   
        local_loss = self.triplet_criterion('local', local_feat, label)
        label = one_hot_smooth_label(label, 20)
        cls_loss = self.cls_criterion(cls_score, label)
        Att_loss = nn.SmoothL1Loss()(Att_M, M)
        parameters = self.get_affine(M)
        Aff_loss = nn.SmoothL1Loss()(Aff_theta, parameters)
        
#         L_dis=self.get_L_dis(embedding,label)
#         L_div=self.get_L_div(embedding)
#         MC_loss=L_dis-10*L_div
        
        
      
        
        total_loss = global_loss + local_loss + cls_loss  + 16 * Att_loss + 16 * Aff_loss

        return total_loss


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
    step_loss=0
    step_metric=0
    with torch.no_grad():
        for images,labels in valid_dl:
            images=images.to(device).float()
            global_feat,local_feat,cls_score,_,_,_=model(images)
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
        if epoch >= 55:
            epoch = epoch - 55
            for param in optimizer.param_groups:
                lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / 5)) / 2
                param['lr'] = lr


def main(train_dl, valid_dl, k):
    criterion = Criterion()
    checkpoint_path = 'base_step{}.pt'.format(k)
    early_stop = EarlyStopping(checkpoint_path)

    model = myNet()
    device = torch.device('cuda:0')
    model = model.to(device)

    epochs = 80
    optimizer = RangerLars(model.parameters(), lr=0.001)
    optimizer_center=torch.optim.SGD(criterion.center_criterion.parameters(),lr=0.5)
    scheduler = flat_and_anneal(epochs)

    for epoch in range(epochs):
        with tqdm(total=len(train_dl)) as pbar:
            train_loss = 0
            steps = len(train_dl)
            for image, labels in train_dl:
                model.train()
                optimizer.zero_grad()

                image = image.to(device).float()
                global_feat, local_feat, cls_score, Att_M, Aff_theta, M = model(image)
                global_feat = global_feat.to('cpu')
                cls_score = cls_score.to('cpu')
                local_feat = local_feat.to('cpu')
                Att_M = Att_M.to('cpu')
                Aff_theta = Aff_theta.to('cpu')
                M = M.to('cpu')
            

                loss = criterion( global_feat, local_feat,cls_score, labels, Att_M, Aff_theta, M)
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
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
#         (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
#         (ShearX, 0., 0.3),
#         (ShearY, 0., 0.3),
#         (TranslateXabs, 0., 100),
#         (TranslateYabs, 0., 100),
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
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img
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
    label_map_image[label].append(name)
    label_set.add(label)
for i in range(test_csv.shape[0]):
    info=test_csv.iloc[i]
    name=info['FileID']
    name=os.path.join(image_dir,name+'.jpg')
    valid_names.append(name)
    label=info['SpeciesID']
    image_labels[name]=label
    label_map_image[label].append(name)
label_set=list(label_set)
rand_augment=RandAugment(3,9)
transform_train=transforms.Compose([
        transforms.Resize([300,300]),

        transforms.RandomHorizontalFlip(),
#         rand_augment,
        transforms.ToTensor()
           ]

)
transform_valid=transforms.Compose([
    transforms.Resize([300,300]),
    transforms.ToTensor()
   
])

k=0

train_label_map_image=defaultdict(list)
for name in train_names:
    name=name
    label=image_labels[name]
    train_label_map_image[label].append(name)
train_ds=TrainDataset(train_names,image_labels,train_label_map_image,transform_train)
valid_ds=TestDataset(valid_names,image_labels,transform_valid)
train_dl=Data.DataLoader(train_ds,batch_size=12,collate_fn=train_collate,shuffle=True,drop_last=True)
valid_dl=Data.DataLoader(valid_ds,batch_size=8,drop_last=True)
main(train_dl,valid_dl,k)
# import copy
# batch_label_map_image = copy.deepcopy(train_label_map_image)
# print(len(train_names))
# batch_names = copy.deepcopy(train_names)
# random.shuffle(batch_names)
# print(len(batch_names))
# batch_index = []
# while True:
#     batches = []
#     for i in range(4):
#         name = batch_names[0]
#         batch = []
#         batch.append(train_names.index(name))
#         batch_names.remove(name)
#         label = image_labels[name]

#         batch_label = list(set(label_set) ^ set([label]))
#         if name in batch_label_map_image[label]:
#             batch_label_map_image[label].remove(name)
#         random.shuffle(batch_label)
#         label_names = batch_label_map_image[label]

#         if len(label_names) == 0:
#             label_names = train_label_map_image[label]
#             name = np.random.choice(label_names)
#         else:
#             name = np.random.choice(label_names)
#             batch_label_map_image[label].remove(name)
#         batch.append(train_names.index(name))

#         label = batch_label[0]

#         label_names = batch_label_map_image[label]
#         if len(label_names) == 0:
#             label_names = train_label_map_image[label]
#             name = np.random.choice(label_names)
#         else:
#             name = np.random.choice(label_names)
#             batch_label_map_image[label].remove(name)
#         if name in batch_names:
#             batch_names.remove(name)
#         index = train_names.index(name)
#         batch.append(index)
#         batches.extend(batch)
#     batch_index.append(batches)
#     if len(batch_names) <= 3:
#         break

# print("origin_size:{},current_size:{}".format(len(train_names) / 21, len(batch_index)))
# print(len(batch_index[-1]))
# train_ds = TestDataset(train_names, image_labels, transform_train)
# valid_ds = TestDataset(valid_names, image_labels, transform_valid)
# sampler = myBatchSampler(batch_index, batch_size=16, drop_last=True)
# train_dl = Data.DataLoader(train_ds, batch_sampler=sampler)
# valid_dl = Data.DataLoader(valid_ds, batch_size=8, drop_last=True)
# k = 0
# main(train_dl, valid_dl, k)
