!pip install efficientnet_pytorch
import os

import cv2

import torch

import pandas as pd 

import numpy as np

from tqdm import tqdm_notebook as tqdm 

import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, Dataset

from efficientnet_pytorch import EfficientNet

from torch.utils.data import SequentialSampler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torchvision import transforms

from torch.utils.data.sampler import *

from torchvision.transforms import *

from sklearn.metrics import roc_auc_score

from torch.nn import *

import torch

import torchvision

import torch.nn as nn

import torch.nn.functional as F

from torch import optim

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

from torch.nn.parallel.data_parallel import data_parallel

from torch.utils.data.sampler import *

from torchvision import transforms
SIZE=128



train_augment = Compose([

    ToPILImage(),

    RandomResizedCrop(SIZE), 

    RandomHorizontalFlip(),

    ToTensor(),

    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])



valid_augment = Compose([

    ToPILImage(),

    RandomResizedCrop(SIZE),    

    RandomHorizontalFlip(),

    ToTensor(),

    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])
def do_valid(net, valid_loader):



    valid_num  = 0

    valid_loss = []



    predicts = []

    truths   = []

    probs = []



    for input, truth, image_id in tqdm(valid_loader):

        input = input.cuda()

        truth = truth.cuda().float()

        with torch.no_grad():

            logit = data_parallel(net, input)

            prob  = torch.sigmoid(logit)



        batch_size = len(image_id)

        valid_num += batch_size



        predicts.append(logit.data.cpu().numpy())

        truths.append(truth.data.cpu().numpy())

        probs.append(prob.data.cpu().numpy())

    assert(valid_num == len(valid_loader.sampler))

    predicts = np.concatenate(predicts).squeeze()

    truths   = np.concatenate(truths).squeeze()

    probs = np.concatenate(probs).squeeze()

    score = roc_auc_score(truths,probs)

    predicts = torch.from_numpy(predicts).cuda()

    truths = torch.from_numpy(truths).cuda()

    valid_loss.append(torch.nn.BCEWithLogitsLoss()(predicts, truths))

    valid_loss.append(score)

    return valid_loss
train = pd.read_csv('../input/kaggledays-china/train.csv',index_col=0)

test = pd.read_csv('../input/kaggledays-china/test.csv',index_col=0)

INPUT_DIR = '/kaggle/input/kaggledays-china'
!mkdir /kaggle/train_images

!cp /kaggle/input/kaggledays-china/train3c/train3c/nonstar/* /kaggle/train_images/

!cp /kaggle/input/kaggledays-china/train3c/train3c/star/* /kaggle/train_images/
class MyDataSet(Dataset):

    def __init__(self, split, augmentation, mode='train'):

        super(MyDataSet, self).__init__()

        self.split = split

        self.ids = split['id'].values

        #self.label = split['is_star'].values

        self.aug = augmentation

        self.mode = mode

        if self.mode in ['train','valid']:

            self.label = split['is_star'].values

        else:

            self.label = np.zeros(split.shape[0])



    

    def __getitem__(self, index):

        img_id = self.ids[index]

        img_label = self.label[index]

        if self.mode in ['train','valid']:

            img = cv2.imread(os.path.join('/kaggle/train_images',img_id+'.png'))

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        elif self.mode in ['test']:

            img = cv2.imread(os.path.join(INPUT_DIR,'test3c','test3c','test3c_data',img_id+'.png'))

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        img = self.aug(img)

        return img,img_label,img_id

        

    def __len__(self):

        return len(self.split)



class EFB_classification(nn.Module):

    def __init__(self,model_name='efficientnet-b0',num_class=1):

        super(EFB_classification, self).__init__()

        self.base = EfficientNet.from_pretrained(model_name)

        in_features = self.base._fc.in_features

        self.base._fc = nn.Linear(in_features=in_features, out_features=num_class)

    

    def forward(self, x):

        return self.base(x)
my_loss = torch.nn.BCEWithLogitsLoss()

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(5,random_state=42,shuffle=True)

net = EFB_classification('efficientnet-b2',num_class=1).cuda()

for train_n,val_n in kf.split(train.is_star.values,train.is_star.values):

    train_n = train.iloc[train_n]

    val_n = train.iloc[val_n]

    break

batch_size = 64

valid_batch_size = 64

train_dataset = MyDataSet(train_n, train_augment, 'train')

train_loader  = DataLoader(

                    train_dataset,

                    sampler     = RandomSampler(train_dataset),

                    batch_size  = batch_size,

                    drop_last   = True,

                    num_workers = 2,

                    pin_memory  = True)



valid_dataset = MyDataSet(val_n, valid_augment, 'valid')

valid_loader  = DataLoader(

                    valid_dataset,

                    sampler     = SequentialSampler(valid_dataset),

                    batch_size  = valid_batch_size,

                    drop_last   = False,

                    num_workers = 2,

                    pin_memory  = True)



optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=0.0003, weight_decay=0.01)

checkpoint = {

    'model': net.state_dict(),

}

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 12 , eta_min=0.00003, last_epoch=-1)



epochs = 12

net.train()

loss_meter = []

for epoch in range(epochs):

    loss_meter = []

    for input, truth, image_id in tqdm(train_loader,'training'):

        optimizer.zero_grad()

        input = input.cuda()

        truth = truth.view(-1,1).float().cuda()

        logit = data_parallel(net, input)

        loss  = my_loss(logit, truth)



        loss.backward()

        optimizer.step()

        loss_meter.append(loss.item())

    train_loss = np.mean(loss_meter)

    net.eval()

    val_sum = do_valid(net, valid_loader)

    val_loss = val_sum[0].item()

    val_auc = val_sum[1]

    loss_str = '_'.join([str(item) for item in [val_loss, val_auc]])

    scheduler.step()

    if True:

        print('validation summmary:EPOCH{}====={}'.format(epoch,loss_str))

        #torch.save(checkpoint,out_dir +'/checkpoint/fold_%d_epoch_%02d_train_loss_%.4f_val_loss_%.4f_val_auc_%.4f_model.pth'%(fold,epoch,train_loss,val_loss,val_auc))

        #print('\n save model to /checkpoint/fold_%d_epoch_%02d_train_loss_%.4f_val_loss_%.4f_val_auc_%.4f_model.pth'%(fold,epoch,train_loss,val_loss,val_auc))



print('Success!')
net.eval()

print('predict test file')
test_df = pd.read_csv('/kaggle/input/kaggledays-china/test.csv')

test_augment = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(int(SIZE*1.2)),

    transforms.TenCrop(SIZE,vertical_flip=True),

    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transforms.ToTensor()(crop)) for crop in crops])),

])

test_dataset = MyDataSet(test_df, test_augment, 'test')

test_loader  = DataLoader(

                    test_dataset,

                    sampler     = SequentialSampler(test_dataset),

                    batch_size  = batch_size//8,

                    drop_last   = False,

                    num_workers = 2,

                    pin_memory  = True)
net.eval()

all_probs,all_id = [],[]

for input, truth, image_id in tqdm(test_loader,'test'):

    bs, ncrops, c, h, w = input.size()

    input = input.cuda()

    truth = truth.cuda()

    with torch.no_grad():

        logit = data_parallel(net, input.view(-1,c,h,w))

        prob  = F.sigmoid(logit)

        prob  = prob.view(bs, ncrops, -1).mean(1)

    prob = prob.squeeze().data.cpu().numpy()

    all_probs.append(prob)

    all_id.extend(image_id)

all_probs = np.hstack(all_probs)
sub = pd.DataFrame({'id':all_id,'is_star':all_probs})

sub.to_csv('submission.csv',index=False,header=True)