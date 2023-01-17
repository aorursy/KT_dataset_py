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
torch.backends.cudnn.benchmark=True
train_paths=get_image_files('/kaggle/input/kernel9ee26a16dd/result')
train_image_infos=[]
label_dict={'bed':0,'sofa':1}
label_names=defaultdict(list)
for path in train_paths:
    info={}
    path=str(path)
    name=path.split('/')[-1].split('.')[0]
  
    label=name.split('_')[0]
    if name.endswith('mask'):
        pass
    else:    
        info['label']=label
        image=cv2.imread(path)
        mask_path=os.path.join('/kaggle/input/kernel9ee26a16dd/result',name+'_mask'+'.png')
        mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
         
        info['image']=image
        info['mask']=mask
        info['name']=name
     
        
        train_image_infos.append(info)
        
test_paths=get_image_files('/kaggle/input/hlw-jj-test/test_data')
test_image_infos=[]
for path in test_paths:
    info={}
    path=str(path)
    name=path.split('/')[-1].split('.')[0]
    
    label=name.split('_')[0]
    if name.endswith('mask'):
        pass
    else:    
        info['label']=label
        image=cv2.imread(path)
        mask_path=os.path.join('/kaggle/input/hlw-jj-test/test_data',name+'_mask'+'.png')
       
        mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
      
        info['image']=image
        info['mask']=mask
        info['name']=name
    
        test_image_infos.append(info)
    
print(len(train_image_infos))
print(len(test_image_infos))

    
class myDataset(Data.Dataset):
    def __init__(self,infos,transform,num_class):
        super(myDataset,self).__init__()
        self.infos=infos
        self.num_class=num_class  
        self.transform=transform
        
    def read_image(self,image,mask):
        augmented=self.transform(image=image,mask=mask)
        new_image=augmented['image']
        new_mask=augmented['mask']
        
        masks=np.zeros((new_mask.shape[0],new_mask.shape[1],self.num_class))
        label=info['label']
        new_mask[np.where(new_mask>0)]=1
        masks[:,:,label_dict[label]]=new_mask
        
        new_image=torch.from_numpy(new_image.transpose((2,0,1)))
        new_mask=torch.from_numpy(masks.transpose((2,0,1)))
        return new_image,new_mask
        
        
    def __getitem__(self,index):
        info=self.infos[index]
        image=info['image']
        mask=info['mask']
    
        
        augmented=self.transform(image=image,mask=mask)
        new_image=augmented['image']
        new_mask=augmented['mask']
        
        masks=np.zeros((new_mask.shape[0],new_mask.shape[1],self.num_class))
        label=info['label']
        new_mask[np.where(new_mask>0)]=1
        masks[:,:,label_dict[label]]=new_mask
        
        new_image=torch.from_numpy(new_image.transpose((2,0,1)))
        new_mask=torch.from_numpy(masks.transpose((2,0,1)))
    
        
        return new_image,new_mask,label_dict[label]
      
        
        
    def __len__(self):
        return len(self.infos)
        
def create_dataloader(infos,transform,batch_size,num_class,shuffle=False):
    dataset=myDataset(infos,transform,num_class)
    dataloader=Data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader
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
train_transform=A.Compose([
    A.Flip(),
    A.ShiftScaleRotate(scale_limit=0),
    A.Transpose(),
    A.OneOf(
        [
            A.IAASharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.6,
    ),

    A.OneOf(
        [
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ],
        p=0.6,
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

])

test_transform=A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader=create_dataloader(train_image_infos,train_transform,6,2,True)
test_loader=create_dataloader(test_image_infos,test_transform,8,2)

class Mask_Criterion:
    def __init__(self):
        self.dice_criterion=smp.utils.losses.DiceLoss()

    def __call__(self,y_pr,y_gt):
        dice_loss=self.dice_criterion(y_pr,y_gt)
        return torch.log(torch.exp(dice_loss)+torch.exp(-1*dice_loss)/2.0)
    
class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2
        self.scale_neg = 50
        self.count=0

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
            if len(pos_pair_)==0:
                print('MS False  count:{}'.format(self.count))
                self.count+=1
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

def one_hot_smooth_label(x, num_class, smooth=0.1):
    num = x.shape[0]
    labels = torch.zeros((num, 2))
    for i in range(num):
        labels[i][x[i]] = 1
    labels = (1 - (num_class - 1) / num_class * smooth) * labels + smooth / num_class
    return labels


class Cls_Criterion:
    def __init__(self):
        self.ms_criterion=MultiSimilarityLoss()
        self.cls_criterion=nn.BCEWithLogitsLoss()
        
    
        
    def __call__(self,logits,label):
  
        label=one_hot_smooth_label(label,2)
        cls_loss=self.cls_criterion(logits,label)
        return cls_loss
        
        
    
        
        

    
class Scheduler:
    def __init__(self,epochs,base_lr=0.001,min_lr=0):
        self.epochs = epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def __call__(self, epoch, optimizer):
        if epoch >= 60:
            epoch = epoch - 60
            for param in optimizer.param_groups:
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

import segmentation_models_pytorch as smp
device=torch.device('cuda:0')
model=smp.DeepLabV3Plus(encoder_name='efficientnet-b3',classes=2,aux_params={'classes':2})
model=model.to(device)


mask_criterion=Mask_Criterion()
cls_criterion=Cls_Criterion()
optimizer=RangerLars(model.parameters(),lr=0.001)
epochs=100
scheduler=Scheduler(epochs)
get_metric=smp.utils.metrics.IoU(threshold=0.75)
callback=EarlyStopping('result.pt')

def evaluate(model,device):
    total_metric=0
    steps=len(test_loader)
    stop=False
    data=[]
    with torch.no_grad():
        for images,masks,labels in test_loader:
            bs=images.shape[0]
            images=images.to(device)
            preds,logits=model(images)
            preds=torch.sigmoid(preds)
            preds=preds.to('cpu')
            
            
            for index in range(bs):
                
                pred=preds[index]
                mask=masks[index]
                metric=get_metric(pred,mask)
                print(metric)
               
                total_metric+=metric
          
            
    total_metric=total_metric/len(test_loader.dataset)
    print('metric:{}'.format(total_metric))
    
    return total_metric

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
from tqdm import tqdm
for epoch in range(epochs):
    with tqdm(total=len(train_loader)) as pbar:
        total_loss=0
        model.train()
        for image,mask,labels in train_loader:
            p=np.random.rand(1)
            
            optimizer.zero_grad()
            image=image.to(device)
            
            if p>0.6:
                lam = np.random.beta(1, 1)
                rand_index = torch.randperm(image.size()[0]).cuda()
#                 w,h=image.shape[1],image.shape[2]
#                 w_cutmix,h_cutmix=int(w*2/3),int(h*2/3)
#                 w_start,h_start=int(np.random.choice(w-w_cutmix-1,1)),int(np.random.choice(h-h_cutmix-1,1))
#                 w_start2,h_start2=int(np.random.choice(w-w_cutmix-1,1)),int(np.random.choice(h-h_cutmix-1,1))
                bbx1,bby1,bbx2,bby2=rand_bbox(image.shape,lam)
                image[:,:,bbx1:bbx2,bby1:bby2]=image[rand_index,:,bbx1:bbx2,bby1:bby2]
                mask[:,:,bbx1:bbx2,bby1:bby2]=mask[rand_index,:,bbx1:bbx2,bby1:bby2]
                labels_a=labels
                labels_b=labels[rand_index]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
                
                pred,logits=model(image)
                pred=torch.sigmoid(pred)
                pred=pred.to('cpu')
                logits=logits.to('cpu')
               
                loss=mask_criterion(pred,mask)
                loss+=lam*cls_criterion(logits,labels_a)+(1-lam)*cls_criterion(logits,labels_b)
                
                
                
            else:    
             
                pred,logits=model(image)
                pred=torch.sigmoid(pred)
                pred=pred.to('cpu')
                logits=logits.to('cpu')
                loss=mask_criterion(pred,mask)+cls_criterion(logits,labels)
            total_loss+=loss
            loss.backward()
            optimizer.step()
            
            pbar.update(1)
        print('epoch:{},loss:{}'.format(epoch,total_loss/len(train_loader)))
        model.eval()
        metric=evaluate(model,device)
        
        callback(metric,model)
        scheduler(epoch,optimizer)
        
            
            
            
