import pandas as pd

import numpy as np

import cv2

import gc

import os

import re

from tqdm import tqdm

from PIL import Image

import albumentations as A

from albumentations.pytorch.transforms import ToTensor



import torch

import torchvision



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator



from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler



from matplotlib import pyplot as plt



df = pd.read_csv('/kaggle/input/face-mask-detection-dataset/train.csv')

df.shape
len(df.classname.value_counts())
df.head()
### Headers Of CSV File Has Error, So Correcting it



df.rename(columns = {'x2' : 'y1', 'y1' : 'x2'}, inplace = True)

df.head()
## Converting The Classname to Respective Integer Label using Sklearn's LabelEncoder



from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

le.fit(df['classname'])

df['classname']=(le.transform(df['classname'])+1)

df.head()
## Checking Whether There Is Any NAN or Missing Values In The DataFrame



df.isnull().sum()
## Splitting The Dataset

### For Final Training Purpose, Use Whole Dataset For Training Rather Than Splitting It Into Valid Set



image_ids = df['name'].unique()

image_ids.sort()

valid_ids=image_ids[:0] 

train_ids=image_ids[:]
valid_df = df[df['name'].isin(valid_ids)]

train_df = df[df['name'].isin(train_ids)]
valid_df.shape, train_df.shape
df.classname.unique()
class Maskdataset(Dataset):



    def __init__(self, dataframe, transforms=None):

        super().__init__()



        self.image_ids = dataframe['name'].unique()

        self.df = dataframe

        self.transforms = transforms



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['name'] == image_id]



        image = cv2.imread('../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'+f'{image_id}', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        boxes = torch.as_tensor(records[['x1', 'y1', 'x2', 'y2']].values, dtype=torch.float32)

        

        



        # there are 21 classes

        labels = torch.as_tensor(records.classname.values,dtype=torch.int64)

        



        keep = (boxes[:, 3]>boxes[:, 1]) & (boxes[:, 2]>boxes[:, 0]) ## To Handle NAN LOSS Cases 

        boxes = boxes[keep]

        labels = labels[keep]



        

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['image_id'] = torch.tensor([index])

        # target['area'] = area



        if self.transforms:

            sample = {

                'image': image,

                'bboxes': target['boxes'],

                'labels': labels

            }

            sample = self.transforms(**sample)

            image = sample['image']

            

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)



        return image, target, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]

def get_train_transform():

    return A.Compose([

        ToTensor()

    ])

        



def get_valid_transform():

    return A.Compose([

        ToTensor()

    ])
## USING SWISH Activation Rather Than The Regular ReLU



import torch.nn as nn



class Swish(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, x):

        return x * torch.sigmoid(x)

    

    

def convert_relu_to_swish(model):

    for child_name, child in model.named_children():

        if isinstance(child, nn.ReLU):

            setattr(model, child_name, Swish())

        else:

            convert_relu_to_swish(child)
## Using Pytorch Faster-RCNN Resnt50 Pretrained Model

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 21  # 20 class (masks) + background



# get number of input features for the classifier

in_features = model.roi_heads.box_predictor.cls_score.in_features



# replace the pre-trained head with a new one

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

convert_relu_to_swish(model) # converting ReLU to SWISH Activation



## Loading The Trained Model Weights

model.load_state_dict(torch.load('../input/face-mask-detection-weights/model-epoch7.pth'))

## To Count The Loss During Training



class Averager:

    def __init__(self):

        self.current_total = 0.0

        self.iterations = 0.0



    def send(self, value):

        self.current_total += value

        self.iterations += 1



    @property

    def value(self):

        if self.iterations == 0:

            return 0

        else:

            return 1.0 * self.current_total / self.iterations



    def reset(self):

        self.current_total = 0.0

        self.iterations = 0.0

## Defining DataLoaders



def collate_fn(batch):

    return tuple(zip(*batch))







train_dataset = Maskdataset(train_df,get_train_transform())

# valid_dataset = Maskdataset(valid_df, get_valid_transform()) ## No need for Final Training Purpose









train_data_loader = DataLoader(

    train_dataset,

    batch_size=4,

    shuffle=True,

    num_workers=4,

    collate_fn=collate_fn

)



# valid_data_loader = DataLoader(

#     valid_dataset,

#     batch_size=8,

#     shuffle=False,

#     num_workers=2,

#     collate_fn=collate_fn

# )

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
images, targets, image_ids = next(iter(train_data_loader))

images = list(image.to(device) for image in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)

sample = images[2].permute(1,2,0).cpu().numpy()
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

ax.imshow(sample)

for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 2)

    

ax.set_axis_off()

ax.imshow(sample)
from torch.optim.optimizer import Optimizer

class Ralamb(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.buffer = [[None, None, None] for ind in range(10)]

        super(Ralamb, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(Ralamb, self).__setstate__(state)



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

                    raise RuntimeError('Ralamb does not support sparse gradients')



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



                # Decay the first and second moment running average coefficient

                # m_t

                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # v_t

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)



                state['step'] += 1

                buffered = self.buffer[int(state['step'] % 10)]



                if state['step'] == buffered[0]:

                    N_sma, radam_step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    # more conservative since it's an approximated value

                    if N_sma >= 5:

                        radam_step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    else:

                        radam_step_size = 1.0 / (1 - beta1 ** state['step'])

                    buffered[2] = radam_step_size



                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)



                # more conservative since it's an approximated value

                radam_step = p_data_fp32.clone()

                if N_sma >= 5:

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    radam_step.addcdiv_(-radam_step_size * group['lr'], exp_avg, denom)

                else:

                    radam_step.add_(-radam_step_size * group['lr'], exp_avg)



                radam_norm = radam_step.pow(2).sum().sqrt()

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                if weight_norm == 0 or radam_norm == 0:

                    trust_ratio = 1

                else:

                    trust_ratio = weight_norm / radam_norm



                state['weight_norm'] = weight_norm

                state['adam_norm'] = radam_norm

                state['trust_ratio'] = trust_ratio



                if N_sma >= 5:

                    p_data_fp32.addcdiv_(-radam_step_size * group['lr'] * trust_ratio, exp_avg, denom)

                else:

                    p_data_fp32.add_(-radam_step_size * group['lr'] * trust_ratio, exp_avg)



                p.data.copy_(p_data_fp32)



        return loss



""" Lookahead Optimizer Wrapper.

Implementation modified from: https://github.com/alphadl/lookahead.pytorch

Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610

"""

import torch

from torch.optim.optimizer import Optimizer

from collections import defaultdict

import math



class Lookahead(Optimizer):

    def __init__(self, base_optimizer, alpha=0.5, k=6):

        if not 0.0 <= alpha <= 1.0:

            raise ValueError(f'Invalid slow update rate: {alpha}')

        if not 1 <= k:

            raise ValueError(f'Invalid lookahead steps: {k}')

        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)

        self.base_optimizer = base_optimizer

        self.param_groups = self.base_optimizer.param_groups

        self.defaults = base_optimizer.defaults

        self.defaults.update(defaults)

        self.state = defaultdict(dict)

        # manually add our defaults to the param groups

        for name, default in defaults.items():

            for group in self.param_groups:

                group.setdefault(name, default)



    def update_slow(self, group):

        for fast_p in group["params"]:

            if fast_p.grad is None:

                continue

            param_state = self.state[fast_p]

            if 'slow_buffer' not in param_state:

                param_state['slow_buffer'] = torch.empty_like(fast_p.data)

                param_state['slow_buffer'].copy_(fast_p.data)

            slow = param_state['slow_buffer']

            slow.add_(group['lookahead_alpha'], fast_p.data - slow)

            fast_p.data.copy_(slow)



    def sync_lookahead(self):

        for group in self.param_groups:

            self.update_slow(group)



    def step(self, closure=None):

        # print(self.k)

        # assert id(self.param_groups) == id(self.base_optimizer.param_groups)

        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:

            group['lookahead_step'] += 1

            if group['lookahead_step'] % group['lookahead_k'] == 0:

                self.update_slow(group)

        return loss



    def state_dict(self):

        fast_state_dict = self.base_optimizer.state_dict()

        slow_state = {

            (id(k) if isinstance(k, torch.Tensor) else k): v

            for k, v in self.state.items()

        }

        fast_state = fast_state_dict['state']

        param_groups = fast_state_dict['param_groups']

        return {

            'state': fast_state,

            'slow_state': slow_state,

            'param_groups': param_groups,

        }



    def load_state_dict(self, state_dict):

        fast_state_dict = {

            'state': state_dict['state'],

            'param_groups': state_dict['param_groups'],

        }

        self.base_optimizer.load_state_dict(fast_state_dict)



        # We want to restore the slow state, but share param_groups reference

        # with base_optimizer. This is a bit redundant but least code

        slow_state_new = False

        if 'slow_state' not in state_dict:

            print('Loading state_dict from optimizer without Lookahead applied.')

            state_dict['slow_state'] = defaultdict(dict)

            slow_state_new = True

        slow_state_dict = {

            'state': state_dict['slow_state'],

            'param_groups': state_dict['param_groups'],  # this is pointless but saves code

        }

        super(Lookahead, self).load_state_dict(slow_state_dict)

        self.param_groups = self.base_optimizer.param_groups  # make both ref same container

        if slow_state_new:

            # reapply defaults to catch missing lookahead specific ones

            for name, default in self.defaults.items():

                for group in self.param_groups:

                    group.setdefault(name, default)





def LookaheadAdam(params, alpha=0.5, k=6, *args, **kwargs):

    adam = Adam(params, *args, **kwargs)

    return Lookahead(adam, alpha, k)



def Over9000(params, alpha=0.5, k=6, *args, **kwargs):

    ralamb = Ralamb(params, *args, **kwargs)

    return Lookahead(ralamb, alpha, k)





RangerLars = Over9000



model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer=Over9000(params,lr=0.0001) ##Over9000 Optimizer (LARS + LookAhead + Ralamb)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.00017, div_factor=2 ,steps_per_epoch=len(train_data_loader), epochs=5)



num_epochs = 25
##COMMENTED FOR SUBMISSION PURPOSE



# import gc

# loss_hist = Averager()





# for epoch in range(num_epochs):

    

#     z=tqdm(train_data_loader)



#     loss_hist.reset()



#     for itr,(images, targets, image_ids) in enumerate(z):

#         torch.cuda.empty_cache()

#         gc.collect()

        

#         images = list(image.to(device).float() for image in images)

#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         loss_dict = model(images, targets)



#         losses = sum(loss for loss in loss_dict.values())

#         loss_value = losses.item()



#         loss_hist.send(loss_value)

#         z.set_description(f'Epoch {epoch+1}/{num_epochs}, LR: %6f, Loss: %.6f'%(optimizer.state_dict()['param_groups'][0]['lr'],loss_value))

#         optimizer.zero_grad()

#         losses.backward()

#         optimizer.step()

#         scheduler.step() ## Since We are using 1-Cycle LR Policy, LR update step has to be taken after every batch





#     print(f"Epoch #{epoch+1} loss: {loss_hist.value}")

#     torch.save(model.state_dict(), f'/content/drive/My Drive/internshala round 1/model-epoch{epoch+1}.pth') 

#     print()

#     print('Saving Model.......')

#     # print()
class MaskTestDataset(Dataset):



    def __init__(self, dataframe, transforms=None):

        super().__init__()



        self.image_ids = dataframe['name'].unique()

        self.df = dataframe

        self.transforms = transforms



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['name'] == image_id]



        image = cv2.imread('../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'+f'{image_id}', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        if self.transforms:

            sample = {

                'image': image,

            }

            sample = self.transforms(**sample)

            image = sample['image']



        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
def get_test_transform():

    return A.Compose([

        ToTensor()

    ])
test_df=pd.read_csv('../input/face-mask-detection-dataset/submission.csv')

test_df.head()
def collate_fn(batch):

    return tuple(zip(*batch))



test_dataset = MaskTestDataset(test_df, get_test_transform())



test_data_loader = DataLoader(

    test_dataset,

    batch_size=2,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)
torch.cuda.empty_cache()

gc.collect()
%%time



detection_threshold = 0.60

results = []

model.eval()

for images, image_ids in test_data_loader:

    torch.cuda.empty_cache()

    gc.collect()



    images = list(image.to(device) for image in images)

    outputs = model(images)



    for i, image in enumerate(images):



        boxes = outputs[i]['boxes'].data.cpu().numpy()

        scores = outputs[i]['scores'].data.cpu().numpy()

        labels = outputs[i]['labels'].data.cpu().numpy()



        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        scores = scores[scores >= detection_threshold]

        image_id = image_ids[i]

        

        

        result = {

            'image_id': image_id,

            'labels': labels,

            'scores': scores,

            'boxes': boxes

        }



        

        results.append(result)
## Using Dictionary is Fastest Way to Create SUBMISSION DATASET.

new=pd.DataFrame(columns=['image_id', 'boxes', 'label'])

rows=[]

for j in range(len(results)):

    for i in range(len(results[j]['boxes'])):

        dict1 = {}

        dict1={"image_id" : results[j]['image_id'],

                  'x1': results[j]['boxes'][i,0],

                  'x2': results[j]['boxes'][i,2],

                  'y1': results[j]['boxes'][i,1],

                  'y2': results[j]['boxes'][i,3],

                  'classname':results[j]['labels'][i].item()}

        rows.append(dict1)





    
sub=pd.DataFrame(rows)

sub['classname']=le.inverse_transform(sub.classname.values - 1) ## Converting Back Labels To Original Names 
sub.head()
sample = images[1].permute(1,2,0).cpu().numpy()

boxes = outputs[1]['boxes'].data.cpu().numpy()

scores = outputs[1]['scores'].data.cpu().numpy()

boxes = boxes[scores >= 0.6].astype(np.int32)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 2)

    

ax.set_axis_off()

ax.imshow(sample)



sub.to_csv('submission.csv', index = False)