import os
import sys

sys.path.append('../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master')
import cv2
import glob
import math
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from contextlib import contextmanager
from tqdm.notebook import tqdm
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import torchvision
from torchvision import models, transforms

from efficientnet_pytorch import model as ENet

import albumentations
from albumentations import RandomResizedCrop, ShiftScaleRotate, Cutout, VerticalFlip, HorizontalFlip
def seed_everything(seed=43):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")
    
    
def load_df(path):
    basename = os.path.basename(path)
    ext = path.split('.')[-1]
    
    if ext == 'csv':
        df = pd.read_csv(path)
    elif ext == 'pkl':
        df = pd.read_pickle(path)
    elif ext == 'parquet':
        df = pd.read_parquet(path)
    else:
        raise IOError(f'Not Accessable Format .{ext}')

    print(f"{basename} shape / {df.shape}")
    
    return df


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='a'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def start(self):        
        self.write('** start training here! **\n')
        self.write('\n')
        self.write('epoch    iter      rate     | train_loss/metric  | valid_loss/metric   | best_epoch/best_score |  min    \n')
        self.write('----------------------------------------------------------------------------------------------------------\n')
        
    def step(self, epoch, niter, lr, train_loss, train_metric, valid_loss, valid_metric, best_epoch, best_metric, time):
        print('\r', end='', flush=True)
        self.write('%5.1f   %5d    %0.6f   | %0.6f  %0.6f | %0.6f   %0.6f |   %6.1f    %6.4f    | %3.1f min \n' % \
                   (epoch, niter, lr, train_loss, train_metric, valid_loss, valid_metric, best_epoch, best_metric, time))

    def flush(self):       
        pass
IMG_SIZE = (128, 128)

class KmnistDataset(Dataset):
    def __init__(self, df, img_base_dir, img_size=IMG_SIZE, transform=None, mode='train'):
        self.df = df
        self.img_size = img_size
        self.transform = transform
        self.mode = mode
        self.img_fnames = self.df['fname'].values
        self.img_num = len(self.img_fnames)
        self.imgs = [self.__load_image(img_base_dir, fname) for fname in tqdm(self.img_fnames)]
     
        print(f'image size {self.img_num}')
        print(f'image num {self.img_num}')
        
    def __load_image(self, img_base_dir, fname):
        img = cv2.imread(os.path.join(img_base_dir, fname), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)

        return img
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        
        img = np.array(img)
        img_origin = img.copy()
        if self.transform is not None:
            img = self.transform(image=img)
            img = img['image']
        
        img = (img / 255.).astype(np.float32)
        img = img[np.newaxis, :, :] 
        img = np.repeat(img, 3, 0)
        
        img_origin = (img_origin / 255.).astype(np.float32)
        img_origin = img_origin[np.newaxis, :, :]
        img_origin = np.repeat(img_origin, 3, 0)
        
        if self.mode in ['train', 'valid']:
            return torch.tensor(img), torch.tensor(img_origin), torch.tensor(self.df.iloc[idx].label)
        else:
            return torch.tensor(img)
        
    def __len__(self):
        return self.img_num
def dataset_loader(df, img_base_dir, transform, mode, sampler, batch_size, num_workers):
    dataset = KmnistDataset(
        df=df, 
        img_base_dir=img_base_dir,
        img_size=(128, 128),
        transform=transform, 
        mode=mode
    )
    
    loader = DataLoader(
        dataset=dataset,
        shuffle=sampler if not sampler else None,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader
pretrained_dict = {f'efficientnet-b{i}': f for i, f in enumerate(sorted(glob.glob('../input/efficientnet-pytorch/*pth')))}

sigmoid = torch.nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
#         i = ctx.saved_variable[0] 
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)

swish_layer = Swish_module()

def relu_fn(x):
    """ Swish activation function """
    return swish_layer(x)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, reduction='mean'):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.5, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.s = s
        self.cos_m = math.cos(m)             #  0.87758
        self.sin_m = math.sin(m)             #  0.47943
        self.th = math.cos(math.pi - m)      # -0.87758
        self.mm = math.sin(math.pi - m) * m  #  0.23971

    def forward(self, logits, labels):
        logits = logits.float()  # float16 to float32 (if used float16)
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # equals to **2
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = DenseCrossEntropy()(output, labels, self.reduction)
        return loss / 2


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class resnet_arcface(nn.Module):
    def __init__(self, out_dim=10):
        super(resnet_arcface, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.gfc = nn.Linear(self.resnet.fc.in_features, 2048)
        self.metric_classify = ArcMarginProduct(2048, out_dim)
        self.myfc = nn.Linear(2048, out_dim)
        self.resnet.fc = nn.Identity()

    def extract(self, x):
        return self.resnet(x)

    def forward(self, x):
        x = self.resnet(x)
        x = Swish_module()(self.gfc(x))
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        metric_output = self.metric_classify(x)
        return out, metric_output


class enet_arcface(nn.Module):
    def __init__(self, backbone='efficientnet-b1', out_dim=10):
        super(enet_arcface, self).__init__()
        self.enet = ENet.EfficientNet.from_name(backbone)
        self.enet.load_state_dict(torch.load(pretrained_dict[backbone]), strict=True)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

        self.gfc = nn.Linear(self.enet._fc.in_features, 4096)
        self.metric_classify = ArcMarginProduct(4096, out_dim)
        self.myfc = nn.Linear(4096, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = Swish_module()(self.gfc(x))
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        metric_output = self.metric_classify(x)
        return out, metric_output
def criterion(logits, metric_logits, target, is_val=False):

    loss = nn.CrossEntropyLoss()(logits, target) 

    if is_val:
        return loss 
    
    loss_metric = ArcFaceLoss()(metric_logits, F.one_hot(target, 10).float()) 
    loss = (loss * 7 + loss_metric * 3) / 10 
        
    return loss

CE = nn.CrossEntropyLoss()
def criterion_mix(logits, metric_logits, target):
    target, shuffled_target, lam = target
    
    loss = nn.CrossEntropyLoss()(logits, target) 
    loss_metric = ArcFaceLoss()(metric_logits, F.one_hot(target, 10).float()) 
    
    loss = (loss * 7 + loss_metric * 3) / 10 
    
    loss_mix = nn.CrossEntropyLoss()(logits, shuffled_target) 
    loss_metric_mix = ArcFaceLoss()(metric_logits, F.one_hot(shuffled_target, 10).float())
    
    loss_mix = (loss_mix * 7 + loss_metric_mix * 3) / 10 

    return lam * loss + (1 - lam) * loss_mix


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

    
def cutmix(data, target, alpha, clip=[0.3, 0.7]):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), clip[0], clip[1])
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return data, targets

def mixup(data, target, alpha, clip=[0.3, 0.7]):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    
    lam = np.clip(np.random.beta(alpha, alpha), clip[0], clip[1])
    data = data * lam + shuffled_data * (1 - lam)
    targets = (target, shuffled_target, lam)
    
    return data, targets
def metric(logits, targets):
    return accuracy_score(targets.cpu().data.numpy(), logits.cpu().data.numpy().argmax(axis=1))

def metric_mix(logits, targets):
    return accuracy_score(targets[0].cpu().data.numpy(), logits.cpu().data.numpy().argmax(axis=1))
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def train(train_loader, model, optimizer, epoch, lr):
    batch_time = AverageMeter()
    losses = AverageMeter()
    metrics = AverageMeter()
    
    model.train()
    
    num_iters = len(train_loader)
    end = time.time()
    iter = 0
    print_freq = 1
    
    for iter, iter_data in enumerate(train_loader, 0):
        batch_time.update(time.time() - end)
            
        images, org_images, targets = iter_data
        images = images.to(DEVICE)
        org_images = org_images.to(DEVICE)
        targets = targets.to(DEVICE)
        
        
        r = np.random.rand(1)
        if r < 0.4:
            images, targets = cutmix(org_images, targets, 1.)
            loss_func = criterion_mix
            metric_func = metric_mix
        elif r < 0.8:
            images, targets = mixup(org_images, targets, 1.)
            loss_func = criterion_mix
            metric_func = metric_mix
        else:
            loss_func = criterion
            metric_func = metric
        
        optimizer.zero_grad()
        logits, metric_logits = model(images)
        loss = loss_func(logits, metric_logits, targets)
        
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item())        
        end = time.time()
        
        metrics.update(metric_func(logits, targets).item())
        
        if (iter + 1) % print_freq == 0 or iter == 0 or (iter + 1) == num_iters:
            print('\r%5.1f   %5d    %0.6f   | %0.4f    %0.4f  | ... ' % \
                 (epoch - 1 + (iter + 1) / num_iters, iter + 1, lr, losses.avg, metrics.avg), end='', flush=True)
            
    return iter, losses.avg, metrics.avg
def validate(valid_loader, model):
    losses = AverageMeter()
    metrics = AverageMeter()
    
    model.eval()
        
    with torch.no_grad():
        for images, org_images, targets in valid_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
        
            logits, metric_logits = model(images)
            loss = criterion(logits, metric_logits, targets, is_val=True)

            losses.update(loss.item())        
            metrics.update(metric(logits, targets).item())

    return losses.avg, metrics.avg
def oof_validate(valid_loader, model):
    model.eval()
    
    preds = []
    
    with torch.no_grad():
        for images, org_images, labels in tqdm(valid_loader):
            images = images.to(DEVICE)
            logits, metric_logits = model(images)
        
            preds.append(logits.cpu().data.numpy())    
    
    preds = np.concatenate(preds)
    return  preds
def predict(test_loader, model):
    model.eval()
    
    preds = []
    for images in test_loader:
        images = images.to(DEVICE)
        logits, _ = model(images)
        
        preds.append(logits.cpu().data.numpy())    
    
    preds = np.concatenate(preds)
    return  preds
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR = os.path.join('..', 'input', 'ailab-ml-training-1')
IMG_BASE_DIR = os.path.join(BASE_DIR, 'train_images', 'train_images')
IMG_TEST_BASE_DIR = os.path.join(BASE_DIR, 'test_images', 'test_images')
seed_everything(seed=43)
train_df = load_df(os.path.join(BASE_DIR, 'train.csv'))
logger = Logger()
logger.open('benchmark')
DEBUG = True
# DEBUG = False 
###############
### settings
###############
n_epochs = 50

##################
### transformer
##################
train_transform = albumentations.Compose([
    RandomResizedCrop(height=IMG_SIZE[0], width=IMG_SIZE[1], scale=(0.9, 1.1), ratio=(0.9, 1.1), p=0.7),
    ShiftScaleRotate(rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=1),
    Cutout(max_h_size=int(IMG_SIZE[0] * 0.85), max_w_size=int(IMG_SIZE[1] * 0.85), num_holes=1, p=0.7),
])
valid_transform = None

############
### KFold
############
kf = StratifiedKFold(n_splits=5, random_state=43, shuffle=True)
oof = np.zeros(len(train_df))
model_state_dicts = []

############
### fold
############
for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df.index, train_df['label'])):
    if fold >= 3:
        break
        
    #############
    ### loader
    #############
    if DEBUG:
        train_loader = dataset_loader(train_df.iloc[train_idx][:1024], IMG_BASE_DIR, train_transform, 'train', RandomSampler(train_df.iloc[train_idx][:1024]), 64, cpu_count())
        valid_loader = dataset_loader(train_df.iloc[valid_idx][:1024], IMG_BASE_DIR, valid_transform, 'valid', None, 64, cpu_count())
    else:
        train_loader = dataset_loader(train_df.iloc[train_idx], IMG_BASE_DIR, train_transform, 'train', RandomSampler(train_df.iloc[train_idx]), 64, cpu_count())
        valid_loader = dataset_loader(train_df.iloc[valid_idx], IMG_BASE_DIR, valid_transform, 'valid', None, 64, cpu_count())

    ############
    ### model
    ############
    model = enet_arcface(backbone='efficientnet-b3')
    model = model.to(DEVICE)

    ##################################
    ### loss, optimizer, scheduler   
    #################################
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    #################
    ### initialize/
    ################
    best_epoch = 0
    start_epoch = 1
    valid_metric = 0
    best_metric = 0
    best_model_state_dict = None

    ################
    ### start log
    ################
    logger.start()

    ######################
    ### train and valid
    ######################
    with timer('train and valid'):
        n_epochs = 1 if DEBUG else n_epochs
        for epoch in range(start_epoch, n_epochs+1):
            end = time.time()
            lr = optimizer.state_dict()['param_groups'][0]['lr'] # refactor

            ############
            ### train
            ############
            iter, train_loss, train_metric = train(train_loader, model, optimizer, epoch, lr)

            #############    
            ### valid
            #############
            valid_loss, valid_metric = validate(valid_loader, model)
            scheduler.step(epoch)

            #############    
            ### save
            #############
            is_best = valid_metric >= best_metric
            if is_best:
                best_epoch = epoch
                best_metric = valid_metric
                best_model_state_dict = model.state_dict()

            ###############    
            ### step log
            ###############
            logger.step(epoch, iter + 1, lr, train_loss, train_metric, valid_loss, valid_metric, best_epoch, best_metric, (time.time() - end) / 60)

    #################
    ### save model
    #################
    torch.save({
        'state_dict': best_model_state_dict,
        'best_epoch': best_epoch,
        'epoch': epoch,
        'best_metric': best_metric,
    }, f'fold_{fold}_epoch_{best_epoch}_best.pth')

    ##########
    ### oof
    ##########
    with torch.no_grad():
        if DEBUG:
            oof[valid_idx[:1024]] = np.argmax(oof_validate(valid_loader, model), axis=1)
        else:
            oof[valid_idx] = np.argmax(oof_validate(valid_loader, model), axis=1)
oof_score = accuracy_score(oof, train_df['label'])
print(oof_score)
sample_submission = load_df(os.path.join(BASE_DIR, 'sample_submission.csv'))
state_dicts = os.listdir('../input/weights-kmnist')
###############
### ensemble
###############

predictions = np.zeros((len(sample_submission), 10))

with timer('predict'):
    #############
    ### loader
    #############
    test_loader = dataset_loader(sample_submission, IMG_TEST_BASE_DIR, None, 'test', None, 64, cpu_count())

    for state_dict in state_dicts:
        model = enet_arcface(backbone='efficientnet-b3')
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(os.path.join('../input/weights-kmnist', state_dict))['state_dict'])
          
        ######################
        ### pred
        ######################
        preds = predict(test_loader, model)
        predictions += preds / len(state_dicts)
predictions = np.argmax(predictions, axis=1).tolist()
sample_submission['label'] = predictions
sample_submission.to_csv('submission.csv', index=False)
from IPython.display import FileLink
FileLink('submission.csv')