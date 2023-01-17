import torch

import torch.utils as utils

import torch.nn as nn

import torch.nn.functional as F



import torch.autograd as autograd

from torch.utils.data.dataset import Dataset

from torch.optim.optimizer import Optimizer, required



import torchvision

import torchvision.utils as vutils

import torchvision.datasets as dset

import torchvision.transforms as transforms

import torchvision.models as models





import math





import os

import argparse

from PIL import Image

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from tqdm import tqdm, tqdm_notebook

import pyarrow

import cv2
parser = argparse.ArgumentParser()

parser.add_argument('--device', default = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))



parser.add_argument('--size', default = 64)

parser.add_argument('--criterion', default = nn.CrossEntropyLoss())

parser.add_argument('--lr', default = 0.0004)

parser.add_argument('--batch_size', default = 32)

parser.add_argument('--Epoch', default = 140)

parser.add_argument('--weight', default = [1, 1/3, 1/5])





parser.add_argument('--name', type = tuple, default = ('grapheme_root','vowel_diacritic', 'consonant_diacritic'))

parser.add_argument('--dim', type = tuple, default = (168, 11, 7))



parser.add_argument('--transform', default = torchvision.transforms.ToTensor())







args, _ = parser.parse_known_args()
train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

data0 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_0.feather')

data1 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_1.feather')

data2 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_2.feather')

data3 = pd.read_feather('/kaggle/usr/lib/resize_and_load_with_feather_format_much_faster/train_data_3.feather')



data_full = pd.concat([data0,data1,data2,data3],ignore_index=True)



class GraphemeDataset(Dataset):

    def __init__(self,df,label,device,transform, _type='train'):

        self.df = df

        self.label = label

        self.device = device

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):

        label1 = self.label.grapheme_root.values[idx]

        label2 = self.label.vowel_diacritic.values[idx]

        label3 = self.label.consonant_diacritic.values[idx]

        

        image = self.transform(255 - self.df.iloc[idx][1:].values.reshape(args.size,args.size).astype(np.float))





        return image,(label1, label2, label3)

    

def make_test_set(loader, device = args.device):

    data = next(iter(loader))

    data[0] = data[0].float()

    data[0] = data[0].to(device = device)

    for d in data[1]:

        d = d.float()

        d = d.to(device = device)

        

    return data
reduced_index =train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(5)).image_id.values

reduced_train = train.loc[train.image_id.isin(reduced_index)]

reduced_data = data_full.loc[data_full.image_id.isin(reduced_index)]

train_image = GraphemeDataset(reduced_data,reduced_train, device = args.device, transform = args.transform)

train_loader = torch.utils.data.DataLoader(train_image,batch_size=30,shuffle=True, num_workers = 4)



fix_data = make_test_set(train_loader)
def accuracy(y, t):

    acc = []

    for y_, t_ in zip(y,t):

        pred_label = torch.argmax(y_, dim=1)

        t_ = t_.to(device = args.device)

        count = pred_label.shape[0]

        correct = (pred_label == t_).sum().float()

        acc_ = correct / count

        acc.append(acc_.item())

    return acc
class ResidualBlock(nn.Module):

    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):

        super(ResidualBlock,self).__init__()

        self.cnn1 =nn.Sequential(

            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(True)

            

        )

        self.cnn2 = nn.Sequential(

            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),

            nn.BatchNorm2d(out_channels)

        )

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),

                nn.BatchNorm2d(out_channels)

            )

        else:

            self.shortcut = nn.Sequential()

            

    def forward(self, x):

        residual = x

        x = self.cnn1(x)

        x = self.cnn2(x)

        x += self.shortcut(residual)

        x = nn.ReLU(True)(x)

        return x
class ResNet34(nn.Module):

    def __init__(self):

        super(ResNet34,self).__init__()

        

        self.block1 = nn.Sequential(

            nn.Conv2d(1,64,kernel_size=2,stride=2,padding=3,bias=False),

            nn.BatchNorm2d(64),

            nn.ReLU(True)

        )

        

        self.block2 = nn.Sequential(

            nn.MaxPool2d(1,1),

            ResidualBlock(64,64),

            ResidualBlock(64,64,2)

        )

        

        self.block3 = nn.Sequential(

            ResidualBlock(64,128),

            ResidualBlock(128,128,2)

        )

        

        self.block4 = nn.Sequential(

            ResidualBlock(128,256),

            ResidualBlock(256,256,2)

        )

        self.block5 = nn.Sequential(

            ResidualBlock(256,512),

            ResidualBlock(512,512,2)

        )

        

        self.avgpool = nn.AvgPool2d(2)

    def dropout(self, x):

        return F.dropout2d(x, 0.1)

        

    def forward(self,x):

        x = self.block1(x)

        x = self.dropout(x)

        x = self.block2(x)

        x = self.dropout(x)

        x = self.block3(x)

        x = self.dropout(x)

        x = self.block4(x)

        x = self.dropout(x)

        x = self.block5(x)

        x = self.dropout(x)

        x = self.avgpool(x)

        x = x.view(x.size(0),-1)



        return x
class Model(nn.Module):

    def __init__(self,args):

        super(Model, self).__init__()

        self.lr = args.lr

        self.dim = args.dim

        self.device = args.device

        self.criterion = args.criterion

        

#        resnet34 = models.resnet34(pretrained = True)

#        cashe = resnet34.conv1.weight[:, 0:1].float()

#        resnet34.conv1 = nn.Conv2d(1, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)

#        resnet34.conv1.weight = nn.Parameter(cashe)





        self.model = ResNet34().to(device = self.device)

        self.init_(self.model)





        



            

        

        self.grapheme = nn.Sequential()

        self.grapheme.add_module('grapheme_classifier', nn.Linear(in_features = 512, out_features = args.dim[0]))

        self.init_(self.grapheme)

#        self.grapheme.add_module('grapheme_sparsemax', Sparsemax())



        

        self.vowel = nn.Sequential()

        self.vowel.add_module('vowel_classifier', nn.Linear(in_features = 512, out_features = args.dim[1]))

        self.init_(self.grapheme)

#        self.vowel.add_module('vowel_sparsemax', Sparsemax())



        self.consonant = nn.Sequential()

        self.consonant.add_module('consonant_classifier', nn.Linear(in_features = 512, out_features = args.dim[2]))

        self.init_(self.consonant)

#        self.consonant.add_module('consonant_sparsemax', Sparsemax())



        



        

        self.model.to(self.device)

        self.grapheme.to(self.device)

        self.vowel.to(self.device)

        self.consonant.to(self.device)





        











    def loss(self, outputs, target, weight = [1, 1, 1]):

        self.l0ss = [self.criterion(outputs[0], target[0].long().to(device = self.device))* weight[0],

                    self.criterion(outputs[1], target[1].long().to(device = self.device))* weight[1],

                    self.criterion(outputs[2], target[2].long().to(device = self.device))* weight[2]]



    

    def backward(self):

        

        (self.l0ss[0] + self.l0ss[1] + self.l0ss[2]).backward()

    

    def zero_grad(self, optim):

        optim.zero_grad()

            

    def step(self, optim):

        optim.step()

        

    

    def forward(self, inputs, training = True):

        inputs = inputs.to(device = self.device)

        if training:

            self.train()

        else:

            self.eval()

        outputs = self.model(inputs)

        grapheme = self.grapheme(outputs)

        vowel = self.vowel(outputs)

        consonant = self.consonant(outputs)

        

        return [grapheme, vowel, consonant, outputs]

    

    def train_(self, inputs, target, optim, training = True):

        self.zero_grad(optim)

#        self.loss(self.forward(inputs), target)

        self.loss(self.forward(inputs, training), target, weight = args.weight)

        self.backward()

        self.step(optim)

        

    def test(self, inputs, target, training = False):

        with torch.no_grad():

            outputs = self.forward(inputs.to(args.device), training = training)

            ac1 = (outputs[0].cpu().argmax(1)==target[0]).float().mean()

            ac2 = (outputs[1].cpu().argmax(1)==target[1]).float().mean()

            ac3 = (outputs[2].cpu().argmax(1)==target[2]).float().mean()

#            print('[{:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(ac1, ac2, ac3, (ac1*2+ac2+ac3)/4))

        return (ac1*2+ac2+ac3)/4

        

        return (ac1*2+ac2+ac3)/4

    def init_(self, model):

        def init_func(m):  # define the initialization function

            classname = m.__class__.__name__

            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

                if init_type == 'normal':

                    init.normal_(m.weight.data, 0.0, init_gain)

                elif init_type == 'xavier':

                    init.xavier_normal_(m.weight.data, gain=init_gain)

                elif init_type == 'kaiming':

                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

                elif init_type == 'orthogonal':

                    init.orthogonal_(m.weight.data, gain=init_gain)

                else:

                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)



                if hasattr(m, 'bias') and m.bias is not None:

                    init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.

                init.normal_(m.weight.data, 1.0, init_gain)

                init.constant_(m.bias.data, 0.0)

            elif classname.find('InstanceNorm2d') != -1:  # InstanceNorm Layer's weight is not a matrix; only normal distribution applies.

                init.normal_(m.weight.data, 1.0, init_gain)

                init.constant_(m.bias.data, 0.0)

        

        return init_func(model)

        

        

        

model = Model(args)

#scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch: (1-epoch/args.Epoch))
class RAdam(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):

        if not 0.0 <= lr:

            raise ValueError("Invalid learning rate: {}".format(lr))

        if not 0.0 <= eps:

            raise ValueError("Invalid epsilon value: {}".format(eps))

        if not 0.0 <= betas[0] < 1.0:

            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))

        if not 0.0 <= betas[1] < 1.0:

            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        

        self.degenerated_to_sgd = degenerated_to_sgd

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):

            for param in params:

                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):

                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])

        super(RAdam, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(RAdam, self).__setstate__(state)



    def step(self, closure=None):



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                state['step'] += 1

                buffered = group['buffer'][int(state['step'] % 10)]

                if state['step'] == buffered[0]:

                    N_sma, step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    if N_sma >= 5:

                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    elif self.degenerated_to_sgd:

                        step_size = 1.0 / (1 - beta1 ** state['step'])

                    else:

                        step_size = -1

                    buffered[2] = step_size



                if N_sma >= 5:

                    if group['weight_decay'] != 0:

                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)

                    p.data.copy_(p_data_fp32)

                elif step_size > 0:

                    if group['weight_decay'] != 0:

                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                    p.data.copy_(p_data_fp32)



        return loss
'''

reduced_index =train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(1)).image_id.values

reduced_train = train.loc[train.image_id.isin(reduced_index)]

reduced_data = data_full.loc[data_full.image_id.isin(reduced_index)]

valid_image = GraphemeDataset(reduced_data,reduced_train, device = args.device, transform = args.transform)

valid_loader = torch.utils.data.DataLoader(valid_image,batch_size=100,shuffle=True, num_workers = 4)



not_valid = train.loc[~train.image_id.isin(reduced_index)]



for epoch in range(args.Epoch):

    

    reduced_index =not_valid.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(5)).image_id.values

    reduced_train = not_valid.loc[not_valid.image_id.isin(reduced_index)]

    reduced_data = data_full.loc[data_full.image_id.isin(reduced_index)]

    train_image = GraphemeDataset(reduced_data,reduced_train, device = args.device, transform = args.transform)

    train_loader = torch.utils.data.DataLoader(train_image,batch_size = args.batch_size,shuffle=True, num_workers = 4)

    

    optim = RAdam(model.parameters(), lr = args.lr * math.cos((epoch/args.Epoch)*math.pi/2) )

    for idx, [data, label] in tqdm(enumerate(train_loader), total = len(train_loader)):

        data = data.float()

        data = data.to(device = args.device)

        model.train_(data, label, optim, True)



    acc = 0

    for idx, [data, label] in tqdm(enumerate(valid_loader), total = len(valid_loader)):

        acc += (model.test(data.float().to(device = args.device), label, False))/len(valid_loader)

    

    print('[{}/{}] : [loss : {:.2f}, {:.2f}, {:.2f}] [acc : {:.1f}%]'.format(epoch+1, args.Epoch, model.l0ss[0],model.l0ss[1],model.l0ss[2], acc*100))

'''
model.load_state_dict(torch.load('/kaggle/input/train-model/ResNet34_1.pth',map_location=args.device))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import cv2

import torch

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms,models

from tqdm import tqdm_notebook as tqdm
class GraphemeDataset_(Dataset):

    def __init__(self,df,device,transform, _type='train'):

        self.df = df

        self.device = device

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):



        

        

        image = self.transform(255 - self.df.iloc[idx][1:].values.reshape(args.size,args.size).astype(np.float))/255



        return image
def Resize(df,size=64):

    resized = {} 

    df = df.set_index('image_id')

    for i in tqdm(range(df.shape[0])):

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized
test_data = ['test_image_data_0.parquet','test_image_data_1.parquet','test_image_data_2.parquet','test_image_data_3.parquet']

predictions = []

batch_size=1

for fname in test_data:

    data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/{fname}')

    data = Resize(data)

    test_image = GraphemeDataset_(data,device = args.device, transform = args.transform)

    test_loader = torch.utils.data.DataLoader(test_image,batch_size=1,shuffle=False)

    with torch.no_grad():

        for idx, (inputs) in tqdm(enumerate(test_loader),total=len(test_loader)):

            inputs.to(args.device)

            inputs = inputs.float()

            outputs1, outputs2, outputs3, _ = model.forward(inputs, training = False)

            

            predictions.append(outputs3.argmax(1).cpu().detach().numpy())

            predictions.append(outputs1.argmax(1).cpu().detach().numpy())

            predictions.append(outputs2.argmax(1).cpu().detach().numpy())



submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')

submission.target = np.hstack(predictions) 

submission.head(100)
submission.to_csv('submission.csv',index=False)