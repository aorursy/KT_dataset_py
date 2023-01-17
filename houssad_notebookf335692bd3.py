# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
import torch.utils.data as Data
from PIL import Image
import numpy as np
import torch
from collections import defaultdict
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
print(torch.__version__)

IMAGE_SIZE=320
train_names = []
test_names = []
image_path = '/kaggle/input/test-birds2/images'
image_label={}
label_map_image=defaultdict(list)
label_id={}
image_id={}
image_tp=defaultdict(list)

with open('/kaggle/input/test-birds2/classes.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        line=line.strip().split(' ')
        id=line[0]
        label=line[1]
        label_id[label]=id

with open('/kaggle/input/test-birds2/images.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        line=line.strip().split(' ')
        id=line[0]
        name=line[1]
        image_id[id]=name

with open('/kaggle/input/test-birds2/train_test_split.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        line=line.strip().split(' ')
        id=line[0]
        tp=line[1]
        image_tp[tp].append(id)
        

        
for id in image_tp['0']:
    image_name=image_id[id]
    path=os.path.join(image_path,image_name)
    train_names.append(path)
    
    label=image_name.split('/')[0]
    label=int(label_id[label])
    image_label[path]=label
    label_map_image[label].append(path)
    
for id in image_tp['1']:
    image_name=image_id[id]
    path=os.path.join(image_path,image_name)
    test_names.append(path)
    
    label=image_name.split('/')[0]
    label=int(label_id[label])
    image_label[path]=label
    
    
        




class myDataset(Data.Dataset):
    def __init__(self,names,transform,image_label,num_class,label_map_images=None,training=True):
        self.names=names
        self.transform=transform
        self.training=training
        self.num_class=num_class
        self.label_map_images=label_map_images
        self.image_label=image_label


    def __len__(self):
        return len(self.names)


    def read_image(self,name):
        image = Image.open(name)
        image = self.transform(image)
        return image

    def __getitem__(self,index):
        name=self.names[index]
        label=self.image_label[name]
        image=self.read_image(name)
        return image,label
      

def create_dataloader(config,batch_size=16,training=True):

    if training:
        names=train_names
        label_map_images=label_map_image
        transform=get_transform('train',config)

    else:
        names=test_names
        label_map_images=None
        transform=get_transform('test',config)
    dataset=myDataset(names,transform,image_label,config['num_class'],label_map_images=label_map_images,training=training)
    dataloader=Data.DataLoader(dataset,batch_size=batch_size)
    return dataloader

def get_transform(type,config):
    if type=='train':
        return transforms.Compose([
            transforms.Resize([config['IMAGE_SIZE'],config['IMAGE_SIZE']]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize([config['TEST_IMAGE_SIZE'],config['TEST_IMAGE_SIZE']]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

!pip install geffnet
!pip install git+https://github.com/zhanghang1989/ResNeSt
import geffnet
import torch.nn as nn
from resnest.torch import resnest50,resnest101,resnest200,resnest269

EPSILON = 1e-12
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
    
    
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix

class WSDAN(nn.Module):
    def __init__(self,config, num_classes, M=32, net_type='b2'):
        super(WSDAN, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.features=get_backbone(net_type)[0]
        self.num_features=get_backbone(net_type)[1]

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)
        self.config=config


    def myforward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
    
        attention_maps = self.attentions(feature_maps)
      
        feature_matrix = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            # Object Localization Am = mean(Ak)
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        # p: (B, self.num_classes)
        # feature_matrix: (B, M * C)
        # attention_map: (B, 2, H, W) in training, (B, 1, H, W) in val/testing

        return p,feature_matrix,attention_map,feature_matrix*100.

    def forward(self, x):
        if self.config['multi_gpu'] and self.config['amp']:
            with torch.cuda.amp.autocast():
                out = self.myforward(x)
                return out
        else:
            out = self.myforward(x)
            return out

        
class CenterLoss(nn.Module):
        def __init__(self):
            super(CenterLoss, self).__init__()
            self.l2_loss = nn.MSELoss(reduction='sum')

        def forward(self, outputs, targets):
            return self.l2_loss(outputs, targets) / outputs.size(0)
net_backbone={
#     'b0':[geffnet.tf_efficientnet_b0_ns(pretrained=True),1280],
#     'b1':[geffnet.tf_efficientnet_b1_ns(pretrained=True),1280],
#     'b2':[geffnet.tf_efficientnet_b2_ns(pretrained=True),1408],
    'b3':[geffnet.tf_efficientnet_b3_ns(pretrained=True),1536],
#     'b4':[geffnet.tf_efficientnet_b4_ns(pretrained=True),1792],
#     'b5':[geffnet.tf_efficientnet_b5_ns(pretrained=True),2048],
#     'b6':[geffnet.tf_efficientnet_b6_ns(pretrained=True),2304],
#     'b7':[geffnet.tf_efficientnet_b7_ns(pretrained=True),2560],
#     'st50':[resnest50(pretrained=True),2048],
#     'st101':[resnest101(pretrained=True),2048],
#     'st200':[resnest200(pretrained=True),2048],
#     'st269':[resnest269(pretrained=True),2048]
}
def mila_(input, beta=1.0):
    return input * torch.tanh(F.softplus(input+beta))


class Mila(nn.Module):

    def __init__(self, beta=1.0):

        super().__init__()
        self.beta = beta

    def forward(self, input):

        return mila_(input, self.beta)

def get_backbone(backbone_type):
    if backbone_type.startswith('st'):
        backbone=net_backbone[backbone_type][0]
        backbone=nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
    else:
        backbone=net_backbone[backbone_type][0]
        backbone=nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            backbone.act1,
            backbone.blocks,
            backbone.conv_head,
            backbone.bn2,
            backbone.act2
        )
    return backbone,net_backbone[backbone_type][1]

import random
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()


    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


!pip install git+https://github.com/pabloppp/pytorch-tools -U
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from collections import defaultdict
from tqdm import tqdm
from torchtools.optim import RangerLars
import torch.nn.functional as F
import math
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  make_pipeline
import pickle

torch.manual_seed(1)
torch.backends.cudnn.benchmark=True

class flat_and_anneal():
    def __init__(self, optimizer,epochs, anneal_start=0.5, base_lr=0.001, min_lr=0):

        self.epochs = epochs
        self.anneal_start = anneal_start
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.optimizer=optimizer
        self.epoch=0

    def step(self):
        epoch=self.epoch
        self.epoch+=1
        if epoch >= int(self.anneal_start*self.epochs):
            epoch = epoch - int(self.anneal_start*self.epochs)
            for param in self.optimizer.param_groups:
                lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / 5)) / 2
                param['lr'] = lr



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

    def __call__(self, val_metric, model,epoch,ema_model=None):
        backup=None
        shadow=None
        if ema_model:
            backup=ema_model.backup
            shadow=ema_model.shadow
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model,backup,shadow)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            print('epoch:{}.best_score:{}'.format(epoch,score))
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model,backup=None,shadow=None):

        state = {'best_metric': metric, 'state': model.state_dict()}
        torch.save(state, self.checkpoint_path)

from fastai.metrics import accuracy
def test_DAN(test_loader,model,device):
    model=model.eval()
    total_acc=0
    with torch.no_grad():
        for image,label in test_loader:
            image=image.to(device)
            label=label.to(device)
            y_pred_raw, _, attention_map,_ = model(image)
            crop_images = batch_augment(image, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop, _, _,_ = model(crop_images)
            y_pred = (y_pred_raw + y_pred_crop) / 2.
            epoch_acc = accuracy(y_pred, label)
            total_acc+=epoch_acc
    return total_acc/len(test_loader)


def train_DAN(config,train_loader,test_loader):
    num_class=config['num_class']
    device=config['device']
    epochs=config['DAN_epochs']
    model=WSDAN(config,num_classes=num_class,M=config['DAN_M'],net_type=config['DAN_type']).to(device)
    feature_center=torch.zeros(num_class,config['DAN_M']*model.num_features).to(device)

    checkpoint_path=config['DAN_ckpt']
    callback=EarlyStopping(checkpoint_path)

    print('Start training DAN ----------')
    if config['multi_gpu'] and torch.cuda.device_count()>1:
        model=nn.DataParallel(model)

    lr=config['DAN_lr']
    if config['DAN_optimizer']=='SGD':
        optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(T_0=10,T_mult=5)
    elif config['DAN_optimizer']=='Rangerlars':
        optimizer=RangerLars(model.parameters(),lr=lr)
        scheduler=flat_and_anneal(optimizer,epochs,0.6,lr)

    if config['amp']:
        scaler=torch.cuda.amp.GradScaler()
    if config['swa']:
        swa_model=WSDAN(config,num_classes=num_class,M=config['DAN_M'],net_type=config['DAN_type']).to(device)
        if config['multi_gpu']:
            swa_model = nn.DataParallel(swa_model)
        swa_n=0

    if config['ema']:
        ema=EMA(model,0.999)
        ema.register()



    entropy_criterion=nn.CrossEntropyLoss()
    center_criterion=CenterLoss()
    for epoch in range(epochs):
        with tqdm(total=len(train_loader)) as pbar:
            model.train()
            for image,label in train_loader:
                optimizer.zero_grad()
                image=image.to(device)
               
                label=label.to(device)

                if config['amp']:
                    with torch.cuda.amp.autocast():
                        y_pred_raw, feature_matrix, attention_map,_ = model(image)
                        feature_center_batch = F.normalize(feature_center[label], dim=-1)
                        feature_center[label] += config['DAN_beta'] * (feature_matrix.detach() - feature_center_batch)
                        with torch.no_grad():
                            crop_images = batch_augment(image, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
                                                        padding_ratio=0.1)
                        y_pred_crop, _, _,_ = model(crop_images)
                        with torch.no_grad():
                            drop_images = batch_augment(image, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))

                        y_pred_drop, _, _,_ = model(drop_images)
                        batch_loss = entropy_criterion(y_pred_raw, label) / 3. + \
                                     entropy_criterion(y_pred_crop, label) / 3. + \
                                     entropy_criterion(y_pred_drop, label) / 3. + \
                                     center_criterion(feature_matrix, feature_center_batch)

                    scaled_grad_params = torch.autograd.grad(scaler.scale(batch_loss), model.parameters(), create_graph=True)
                    inv_scale = 1. / scaler.get_scale()
                    grad_params = [p * inv_scale for p in scaled_grad_params]
                    with torch.cuda.amp.autocast():
                        grad_norm = 0
                        for grad in grad_params:
                            grad_norm += grad.pow(2).sum()
                        grad_norm = grad_norm.sqrt()
                        batch_loss = batch_loss + grad_norm
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.step()

                    pbar.update(1)
                else:
                    y_pred_raw, feature_matrix, attention_map,_ = model(image)
                    feature_center_batch = F.normalize(feature_center[label], dim=-1)
                    feature_center[label] += config['DAN_beta'] * (feature_matrix.detach() - feature_center_batch)
                    with torch.no_grad():
                        crop_images = batch_augment(image, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
                                                    padding_ratio=0.1)
                    y_pred_crop, _, _ ,_= model(crop_images)
                    with torch.no_grad():
                        drop_images = batch_augment(image, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))

                    y_pred_drop, _, _ = model(drop_images)
                    batch_loss = entropy_criterion(y_pred_raw, label) / 3. + \
                                 entropy_criterion(y_pred_crop, label) / 3. + \
                                 entropy_criterion(y_pred_drop, label) / 3. + \
                                 center_criterion(feature_matrix, feature_center_batch)
                    batch_loss.backward()
                    optimizer.step()
                    pbar.update(1)

        if config['ema']:
            ema.update()


        if config['swa'] and (epoch+1)>=config['swa_start'] and (epoch+1-config['swa_start'])% config['swa_c_epochs']==0:
            moving_average(swa_model,model,1.0/(swa_n+1))
            swa_n+=1

        scheduler.step()

        metric=test_DAN(test_loader,model,device)
        print('epoch:{},DAN_metric:{}'.format(epoch))
        callback(metric,model,epoch)


    model.load_state_dict(torch.load(checkpoint_path)['state'])
    metric=test_DAN(test_loader,model,device)
    print('finnal-eval-metric:{}'.format(metric))

    if config['swa']:
        swa_train_loader=create_dataloader(config,config['swa_batch_size'],training=True)
        bn_update(swa_train_loader,swa_model)
        metric=test_DAN(test_loader,swa_model,device)
        print('swa-final-eval-metric:{}'.format(metric))
        torch.save(swa_model.state_dict(),'DAN_swa_final.pt')
    if config['ema']:
        ema.apply_shadow()
        ema_model=ema.model
        metric=test_DAN(test_loader,ema_model,device)
        print('ema-final-eval-metric:{}'.fromat(metric))
        torch.save(ema_model.state_dict(),'DAN_ema_final.pt')


    if config['SVM']:


        global_svm = make_pipeline(StandardScaler(),
                                       LinearSVC(random_state=0, tol=1e-5))




        global_X = []
        global_y = []


        model.load_state_dict(torch.load(checkpoint_path)['state'])
        model.eval()
        for image, label in train_loader:
            with torch.no_grad():
                image = image.to(device)
                if config['amp']:
                    with torch.cuda.amp.autocast():
                        _,_,_,feature_matrix = model(image)
                else:
                    _,_,_,feature_matrix = model(image)
                global_X.append(feature_matrix)
                global_y.append(label.numpy())


        global_X = np.concatenate(global_X, axis=0)
        global_y = np.concatenate(global_y, axis=0)
        global_svm.fit(global_X, global_y)

        total_acc = 0
        for image, label in test_loader:
            label = label.numpy()
            image = image.to(device)
            with torch.no_grad():
                _,_,_,feature_matrix = model(image)

                y = global_svm.predict(feature_matrix.numpy())
                num = y == label
                acc = np.sum(num) / image.shape[0]
                total_acc += acc

        print('DAN_SVM acc:{}'.format(total_acc / len(test_loader)))
        f=open('DAN_svm.pkl','wb')
        pickle.dump(global_svm, f)
        f.close()


    return model


config = {

    'SVM': False,

    'DAN': True,
    'DAN_epochs': 100,
    'DAN_M': 32,
    'DAN_type': 'b3',
    'DAN_ckpt': 'DAN_best.pt',
    'DAN_lr':0.001,
    'DAN_optimizer': 'Rangerlars',
    'DAN_beta': 5e-3,

    'Locate': True,
    'ATF': True,

    'share_backbone': False,

    'MC_loss': False,

    'Triplet': True,

    'MultiSmiliar': False,

    'local': False,

    'multi_patch': False,

    'local_LSTM': False,

    'GridMask': False,

    'multi_gpu': False,

    'amp': True,

    'IMAGE_SIZE': 320,

    'TEST_IMAGE_SIZE': 320,

    'Fix': False,

    'epochs': 100,

    'ckpt': 'best.pt',

    'lr': 0.001,

    'optimizer': 'Rangerlars',

    'device': torch.device('cuda:0'),

    'num_class': 200,

    'batch_size': 32,

    'swa':False,

    'swa_batch_size':16,

    'swa_start':70,

    'swa_c_epochs':2,

    'ema':False,



}
train_loader=create_dataloader(config,config['batch_size'],training=True)
test_loader=create_dataloader(config,config['batch_size'],training=False)

if config['DAN']:
        DAN=train_DAN(config,train_loader,test_loader)
        torch.cuda.empty_cache()
        print('Ending training DAN--------')
