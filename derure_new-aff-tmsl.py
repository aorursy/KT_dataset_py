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
    def __init__(self,k=4):
        super(k_max_pool,self).__init__()
        self.k=k
        self.pool=nn.AdaptiveAvgPool1d(1)
        
    def forward(self,x):
        x=x.view(x.shape[0],x.shape[1],-1)
        x=x.topk(self.k,dim=-1).values
        x=self.pool(x)
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
        pool=k_max_pool()
        swish=mySwish()
        block_0=model.layer1[0]
        block_0.relu=swish
        block_1=model.layer1[1]
        block_1.relu=swish
        block_2=model.layer1[2]
        block_2.relu=swish
        self.model = nn.Sequential(
            model.conv1,
            model.bn1,
            act1,
            model.maxpool,
            block_0,
            block_1,
            block_2
        )

    def forward(self,x):
      
        x = F.interpolate(x, size=(56, 56), mode='bilinear')
     
        x=self.model(x)
        x=torch.mean(x,dim=1).unsqueeze(1)
        return x
    
import torch
import torch.nn as nn
import numpy as np
class MaxLayer(nn.Module):
    def __init__(self):
        super(MaxLayer, self).__init__()

    def forward(self,x):
        x=x.view(x.shape[0],-1)
        x_max=torch.max(x,dim=1).values
        return x_max


class MinLayer(nn.Module):
    def __init__(self):
        super(MinLayer, self).__init__()

    def forward(self,x):
        x=x.view(x.shape[0],-1)
        x_min=torch.min(x,dim=1).values
        return x_min

class TrainSubtractionLayer(nn.Module):
    def __init__(self,sub=0.3):
        super(TrainSubtractionLayer, self).__init__()
        sub=np.array([sub])
        sub=torch.from_numpy(sub)[0].float()
        self.sub=nn.Parameter(sub,requires_grad=True)

    def forward(self,x):

        return x-self.sub

class DivisionLayer(nn.Module):
    def __init__(self):
        super(DivisionLayer, self).__init__()

    def forward(self,x,divisor):
        return x/divisor


class Min_Max_Normalize(nn.Module):
    def __init__(self):
        super(Min_Max_Normalize, self).__init__()
        self.min_layer=MinLayer()
        self.max_layer=MaxLayer()
        self.divisor=DivisionLayer()


    def forward(self,x):
        x_max=self.max_layer(x).view(x.shape[0],1,1,1)
        x_min=self.min_layer(x).view(x.shape[0],1,1,1)
       
        x=self.divisor(x-x_min,x_max-x_min)
        return x

class ThresholdLayer(nn.Module):
    def __init__(self):
        super(ThresholdLayer, self).__init__()
        self.sub_layer=TrainSubtractionLayer()
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        return self.relu(self.sub_layer(x))


class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        self.fc1=nn.Linear(14,128)
        self.relu=mySwish()
        self.fc2=nn.Linear(128,3)
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1,0,0],dtype=torch.float))

    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x
    
class Affnet(nn.Module):
    def __init__(self):
        super(Affnet, self).__init__()
        min_max_normalize=Min_Max_Normalize()
        threshold_layer=ThresholdLayer()
        self.preprocess_model=nn.Sequential(
             min_max_normalize,
             threshold_layer
        )
        self.pool1=nn.MaxPool2d((1,14))
        self.pool2=nn.MaxPool2d((14,1))
        self.sub1=SubModel()
        self.sub2=SubModel()
       
        
    def forward(self,x):
        x=self.preprocess_model(x)
        
        xv=self.pool2(x).squeeze()
        xv=self.sub2(xv)
        xh=self.pool1(x).squeeze()
        xh=self.sub1(xh)
        theta=torch.zeros((x.shape[0],6),device=x.device)
        theta[:,0]=xh[:,0]
        theta[:,3]=xv[:,0]
        theta[:,1]=xh[:,1]
        theta[:,4]=xv[:,1]
        theta[:,2]=xh[:,2]
        theta[:,5]=xh[:,2]
        return theta


def compute_window_nums(ratios, stride, input_size):
    size = input_size / stride
    window_nums = []

    for _, ratio in enumerate(ratios):
        window_nums.append(int((size - ratio[0]) + 1) * int((size - ratio[1]) + 1))

    return window_nums


def nms(scores_np, proposalN, iou_threshs, coordinates):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0]
    indices_coordinates = np.concatenate((scores_np, coordinates), 1)

    indices = np.argsort(indices_coordinates[:, 0])
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0,windows_num).reshape(windows_num,1)), 1)[indices]                  #[339,6]
    indices_results = []

    res = indices_coordinates

    while res.any():
        indice_coordinates = res[-1]
        indices_results.append(indice_coordinates[5])

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1,proposalN).astype(np.int)
        res = res[:-1]

        # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])
        lengths = end_min - start_max + 1
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)
        res = res[iou_map_cur <= iou_threshs]

    while len(indices_results) != proposalN:
        indices_results.append(indice_coordinates[5])

    return np.array(indices_results).reshape(1, -1).astype(np.int)


def ComputeCoordinate(image_size, stride, indice, ratio):
    size = int(image_size / stride)
    column_window_num = (size - ratio[1]) + 1
    x_indice = indice // column_window_num
    y_indice = indice % column_window_num
    x_lefttop = x_indice * stride - 1
    y_lefttop = y_indice * stride - 1
    x_rightlow = x_lefttop + ratio[0] * stride
    y_rightlow = y_lefttop + ratio[1] * stride
    # for image
    if x_lefttop < 0:
        x_lefttop = 0
    if y_lefttop < 0:
        y_lefttop = 0
    coordinate = np.array((x_lefttop, y_lefttop, x_rightlow, y_rightlow)).reshape(1, 4)

    return coordinate

def indices2coordinates(indices, stride, image_size, ratio):
    batch, _ = indices.shape
    coordinates = []

    for j, indice in enumerate(indices):
        coordinates.append(ComputeCoordinate(image_size, stride, indice, ratio))

    coordinates = np.array(coordinates).reshape(batch,4).astype(int)       # [N, 4]
    return coordinates

input_size = 448
stride = 32
N_list = [2, 2]
proposalN = sum(N_list)  # proposal window num
window_side = [128,  256]
iou_threshs = [0.25,  0.25]
ratios = [[4, 4], [3, 5], [5, 3],
          [8, 8], [6, 10], [10, 6], [7, 9], [9, 7], [7, 10], [10, 7]]

window_nums = compute_window_nums(ratios, stride, input_size)
window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:])]
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)

class APPM(nn.Module):
    def __init__(self):
        super(APPM, self).__init__()
        self.avgpools = [nn.AvgPool2d(ratios[i], 1) for i in range(len(ratios))]

    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs, DEVICE='cuda'):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]

        # feature map sum
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))]

        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1)
        windows_scores_np = all_scores.data.cpu().numpy()
        window_scores = torch.from_numpy(windows_scores_np).to(DEVICE).reshape(batch, -1)

        # nms
        proposalN_indices = []
        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j in range(len(window_nums_sum) - 1):
                indices_results.append(
                    nms(scores[sum(window_nums_sum[:j + 1]):sum(window_nums_sum[:j + 2])], proposalN=N_list[j],
                        iou_threshs=iou_threshs[j],
                        coordinates=coordinates_cat[sum(window_nums_sum[:j + 1]):sum(window_nums_sum[:j + 2])]) + sum(
                        window_nums_sum[:j + 1]))
            # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1))  # reverse

        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).to(DEVICE)
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in
             enumerate(all_scores)], 0).reshape(
            batch, proposalN)

        return proposalN_indices, proposalN_windows_scores, window_scores


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


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
            act2
        )
        self.rawcls_net = nn.Linear(1536, 20)
        self.APPM = APPM()

        self.global_avgpool = torch.nn.AdaptiveAvgPool2d(1)
        #         self.global_bn = nn.BatchNorm1d(1536)
        #         self.global_bn.bias.requires_grad = False
        #         self.local_conv = nn.Conv2d(1536, 512, 1)
        #         self.local_bn = nn.BatchNorm2d(512)
        #         self.local_bn.bias.requires_grad = False
        nn.init.kaiming_normal_(self.rawcls_net.weight, mode='fan_out')
        nn.init.constant_(self.rawcls_net.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
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
        local_fm = self.backbone(x)
        M=torch.mean(local_fm,dim=1).unsqueeze(1)
        local_embedding = self.global_avgpool(local_fm)
        local_embedding = local_embedding.view(local_embedding.shape[0], -1)
        local_embedding = F.dropout(local_embedding)
        
        local_logits = self.rawcls_net(local_embedding)
        del local_embedding
        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.APPM(proposalN, local_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs, x.device)
        if self.training:
            window_imgs = torch.zeros([batch_size, proposalN, 3, 224, 224]).to(x.device)  # [N, 4, 3, 224, 224]
            for i in range(batch_size):
                for j in range(proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    window_imgs[i:i + 1, j] = F.interpolate(x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)],
                                                            size=(224, 224),
                                                            mode='bilinear',
                                                            align_corners=True)  # [N, 4, 3, 224, 224]

            window_imgs = window_imgs.reshape(batch_size * proposalN, 3, 224, 224)  # [N*4, 3, 224, 224]
            window_fm = self.backbone(window_imgs.detach())  # [N*4, 2048]
            window_embedding = self.global_avgpool(window_fm)
            window_embedding = window_embedding.view(window_embedding.shape[0], -1)
            window_embeddings = F.dropout(window_embedding)
            proposalN_windows_logits = self.rawcls_net(window_embeddings)  # [N* 4, 200]
        else:
            proposalN_windows_logits = torch.zeros([batch_size * proposalN, 20]).to(x.device)

        return proposalN_windows_logits, local_logits, Att_M, Aff_theta, M


def one_hot_smooth_label(x,num_class,smooth=0.1):
    num=x.shape[0]
    labels=torch.zeros((num,20))
    for i in range(num):
        labels[i][x[i]]=1
    labels=(1-(num_class-1)/num_class*smooth)*labels+smooth/num_class
    return labels

from skimage import measure
    
class Criterion:
    def __init__(self):
    
        self.cls_criterion = nn.BCEWithLogitsLoss()
    
        
    def normalize_transforms(self,transforms, W,H):
            transforms[0,0] = transforms[0,0]
            transforms[0,1] = transforms[0,1]*H/W
            transforms[0,2] = transforms[0,2]*2/W + transforms[0,0] + transforms[0,1] - 1

            transforms[1,0] = transforms[1,0]*W/H
            transforms[1,1] = transforms[1,1]
            transforms[1,2] = transforms[1,2]*2/H + transforms[1,0] + transforms[1,1] - 1

            return transforms
        
    def get_affine(self,M):
        M=(M>0).float()
        parameters=torch.zeros((M.shape[0],6))
        for i,m in enumerate(M):
            m=m.squeeze().numpy()
            component=measure.label(m,connectivity=2)
            prob=measure.regionprops(component)
            miny,minx,maxy,maxx=prob[0].bbox
            miny=(miny)*32-1
            minx=(minx)*32-1
            maxx=(maxx)*32-1
            maxy=(maxy)*32-1
            point1=np.float32([[minx,miny],[maxx,maxy],[minx,maxy]])
            point2=np.float32([[0,0],[448,448],[0,448]])
            
            M=cv2.getAffineTransform(point1,point2)
            a=torch.ones((3,3))
            a[0][0]=M[0][0]
            a[1][0]=M[0][1]
            a[2][0]=M[0][2]
            a[0][1]=M[1][0]
            a[1][1]=M[1][1]
            a[2][1]=M[1][2]
            a[0][2]=0
            a[1][2]=0
            a[2][2]=1
            a=a.inverse()
            m=np.array([
                [a[0][0],a[1][0],a[2][0]],
                [a[0][1],a[1][1],a[2][1]]
            ])
            m=self.normalize_transforms(m,448,448)
            theta=torch.from_numpy(m).float()
            parameters[i]=theta.view(-1,6)[0]
        return parameters
    

    def __call__(self,window_logits,local_logits,label,Att_M,Aff_theta,M):
        Att_loss=nn.SmoothL1Loss()(Att_M,M)
        parameters=self.get_affine(M)
        Aff_loss=nn.SmoothL1Loss()(Aff_theta,parameters)
        window_label= label.unsqueeze(1).repeat(1, proposalN).view(-1)
        window_label=one_hot_smooth_label(window_label,20)
        windowscls_loss = self.cls_criterion(window_logits,window_label
                              )
        label=one_hot_smooth_label(label,20)
        local_loss=self.cls_criterion(local_logits,label)
    
    
        return 16*Att_loss+16*Aff_loss+local_loss+windowscls_loss


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
            _,cls_score,_,_,_=model(images)
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
        if epoch >= 5:
            epoch = epoch - 5
            for param in optimizer.param_groups:
                lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / 5)) / 2
                param['lr'] = lr


def main(train_dl, valid_dl, k):
    criterion = Criterion()
    checkpoint_path = 'step{}.pt'.format(k)
    early_stop = EarlyStopping(checkpoint_path)

    model = myNet()
    device = torch.device('cuda:0')
    model = model.to(device)
    model.load_state_dict(torch.load('step0.pt')['state'])
   
    epochs = 20
    optimizer = RangerLars(model.parameters(), lr=0.001)
    scheduler = flat_and_anneal(epochs)

    for epoch in range(epochs):
        with tqdm(total=len(train_dl)) as pbar:
            train_loss = 0
            steps = len(train_dl)
            for image, labels in train_dl:
                model.train()
                optimizer.zero_grad()

                image = image.to(device).float()
                window_logits,local_logits,Att_M,Aff_theta,M= model(image)
                window_logits=window_logits.to('cpu')
                local_logits=local_logits.to('cpu')
                Att_M=Att_M.to('cpu')
                Aff_theta=Aff_theta.to('cpu')
                M=M.to('cpu')
                loss = criterion(window_logits,local_logits,labels,Att_M,Aff_theta,M)
                train_loss += loss
                loss.backward()

                optimizer.step()

                pbar.update(1)
            print('epoch:{},train_loss:{}'.format(epoch, train_loss / steps))
            model.eval()
            metric = evaluate(model, valid_dl)
            early_stop(metric, model)
            scheduler(epoch, optimizer)


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

import copy
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

k=0

train_label_map_image=defaultdict(list)
for name in train_names:
    name=name
    label=image_labels[name]
    train_label_map_image[label].append(name)
    
batch_label_map_image=copy.deepcopy(train_label_map_image)
batch_names=copy.deepcopy(train_names)
batch_index=[]
while True:
    batches=[]
    for i in range(2):
        name = batch_names[0]
        batch = []
        batch.append(train_names.index(name))
        batch_names.remove(name)
        label = image_labels[name]
        
       
        batch_label = list(set(label_set) ^ set([label]))
        if name in batch_label_map_image[label]:
            batch_label_map_image[label].remove(name)
        random.shuffle(batch_label)
        label_names=batch_label_map_image[label]
        
        if len(label_names) == 0:
            label_names = train_label_map_image[label]
            name = np.random.choice(label_names)
        else:
                name = np.random.choice(label_names)
                batch_label_map_image[label].remove(name)
        batch.append(train_names.index(name))
        
        


       
        label = batch_label[0]
  
        label_names = batch_label_map_image[label]
        if len(label_names) == 0:
            label_names = train_label_map_image[label]
            name = np.random.choice(label_names)
        else:
            name = np.random.choice(label_names)
            batch_label_map_image[label].remove(name)
        if name in batch_names:
                batch_names.remove(name)
        index = train_names.index(name)
        batch.append(index)
        batches.extend(batch)
    batch_index.append(batches)
    if len(batch_names)<=1:
        break
        
        

        
print("origin_size:{},current_size:{}".format(len(train_names)/6,len(batch_index)))
print(len(batch_index[-1]))
train_ds=TestDataset(train_names,image_labels,transform_train)
valid_ds=TestDataset(valid_names,image_labels,transform_valid)
sampler=myBatchSampler(batch_index,batch_size=16,drop_last=True)
train_dl=Data.DataLoader(train_ds,batch_sampler=sampler)
valid_dl=Data.DataLoader(valid_ds,batch_size=8,drop_last=True)
k=0
main(train_dl,valid_dl,k)
# train_ds=TrainDataset(train_names,image_labels,train_label_map_image,transform_train)
# valid_ds=TestDataset(valid_names,image_labels,transform_valid)
# train_dl=Data.DataLoader(train_ds,batch_size=2,collate_fn=train_collate,shuffle=True,drop_last=True)
# valid_dl=Data.DataLoader(valid_ds,batch_size=8,drop_last=True)
# main(train_dl,valid_dl,k)
