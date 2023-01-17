import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2

import os

from tqdm import tqdm,trange

from sklearn.model_selection import train_test_split

import sklearn.metrics



import torch

import torch.nn as nn

import torch.nn.functional as F



import warnings

warnings.filterwarnings("ignore")

import re

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.utils.checkpoint as cp

from collections import OrderedDict

from torch import Tensor

from torch.jit.annotations import List
df_trains = []

df_vals = []

for i in range(46):

    df = pd.read_json(f'../input/deepfake/metadata{i}.json')

    df_trains.append(df)

for i in range (47,49,1) :

    df = pd.read_json(f'../input/deepfake/metadata{i}.json')

    df_vals.append(df)

    

nums = list(range(len(df_trains)+1))

LABELS = ['REAL','FAKE']

val_nums=[47, 48, 49]    
def get_path(num,x):

    num=str(num)

    if len(num)==2:

        path='../input/deepfake/DeepFake'+num+'/DeepFake'+num+'/' + x.replace('.mp4', '') + '.jpg'

    else:

        path='../input/deepfake/DeepFake0'+num+'/DeepFake0'+num+'/' + x.replace('.mp4', '') + '.jpg'

    if not os.path.exists(path):

       raise Exception

    return path

paths=[]

y=[]

for df_train,num in tqdm(zip(df_trains,nums),total=len(df_trains)):

    images = list(df_train.columns.values)

    for x in images:

        try:

            paths.append(get_path(num,x))

            y.append(LABELS.index(df_train[x]['label']))

        except Exception as err:

            #print(err)

            pass



val_paths=[]

val_y=[]

for df_val,num in tqdm(zip(df_vals,val_nums),total=len(df_vals)):

    images = list(df_val.columns.values)

    for x in images:

        try:

            val_paths.append(get_path(num,x))

            val_y.append(LABELS.index(df_val[x]['label']))

        except Exception as err:

            #print(err)

            pass
def read_img(path):

    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)



def shuffle(X,y):

    new_train=[]

    for m,n in zip(X,y):

        new_train.append([m,n])

    random.shuffle(new_train)

    X,y=[],[]

    for x in new_train:

        X.append(x[0])

        y.append(x[1])

    del new_train    

    return X,y



import random

def get_random_sampling(paths, y, val_paths, val_y):

  real=[]

  fake=[]

  for m,n in zip(paths,y):

      if n==0:

          real.append(m)

      else:

          fake.append(m)

  # fake=random.sample(fake,len(real))

  paths,y=[],[]

  for x in real:

      paths.append(x)

      y.append(0)

  for x in fake:

      paths.append(x)

      y.append(1)



  real=[]

  fake=[]

  for m,n in zip(val_paths,val_y):

      if n==0:

          real.append(m)

      else:

          fake.append(m)

  # fake=random.sample(fake,len(real))

  val_paths,val_y=[],[]

  for x in real:

      val_paths.append(x)

      val_y.append(0)

  for x in fake:

      val_paths.append(x)

      val_y.append(1)



  X=[]

  for img in tqdm(paths):

      X.append(read_img(img))

  val_X=[]

  for img in tqdm(val_paths):

      val_X.append(read_img(img))





  X, y = shuffle(X,y)

  val_X, val_y = shuffle(val_X,val_y)



  return X, val_X, y, val_y
from torch.utils.data import Dataset, DataLoader

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]



class ImageDataset(Dataset):

    def __init__(self, X, y, training=True, transform=None):

        self.X = X

        self.y = y

        self.transform = transform

        self.training = training



    def __len__(self):

        return len(self.X)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

        

        img = self.X[idx]



        if self.transform is not None:

          res = self.transform(image=img)

          img = res['image']

        

        img = np.rollaxis(img, 2, 0)

        # img = np.array(img).astype(np.float32) / 255.



        labels = self.y[idx]

        labels = np.array(labels).astype(np.float32)

        return [img, labels]
try:

    from torch.hub import load_state_dict_from_url

except ImportError:

    from torch.utils.model_zoo import load_url as load_state_dict_from_url
__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']



model_urls = {

    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',

    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',

    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',

    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',

}
class _DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):

        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),

        self.add_module('relu1', nn.ReLU(inplace=True)),

        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *

                                           growth_rate, kernel_size=1, stride=1,

                                           bias=False)),

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),

        self.add_module('relu2', nn.ReLU(inplace=True)),

        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,

                                           kernel_size=3, stride=1, padding=1,

                                           bias=False)),

        self.drop_rate = float(drop_rate)

        self.memory_efficient = memory_efficient



    def bn_function(self, inputs):

        # type: (List[Tensor]) -> Tensor

        concated_features = torch.cat(inputs, 1)

        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484

        return bottleneck_output



    # todo: rewrite when torchscript supports any

    def any_requires_grad(self, input):

        # type: (List[Tensor]) -> bool

        for tensor in input:

            if tensor.requires_grad:

                return True

        return False



    @torch.jit.unused  # noqa: T484

    def call_checkpoint_bottleneck(self, input):

        # type: (List[Tensor]) -> Tensor

        def closure(*inputs):

            return self.bn_function(*inputs)



        return cp.checkpoint(closure, input)



    @torch.jit._overload_method  # noqa: F811

    def forward(self, input):

        # type: (List[Tensor]) -> (Tensor)

        pass



    @torch.jit._overload_method  # noqa: F811

    def forward(self, input):

        # type: (Tensor) -> (Tensor)

        pass



    # torchscript does not yet support *args, so we overload method

    # allowing it to take either a List[Tensor] or single Tensor

    def forward(self, input):  # noqa: F811

        if isinstance(input, Tensor):

            prev_features = [input]

        else:

            prev_features = input



        if self.memory_efficient and self.any_requires_grad(prev_features):

            if torch.jit.is_scripting():

                raise Exception("Memory Efficient not supported in JIT")



            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)

        else:

            bottleneck_output = self.bn_function(prev_features)



        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:

            new_features = F.dropout(new_features, p=self.drop_rate,

                                     training=self.training)

        return new_features





class _DenseBlock(nn.ModuleDict):

    _version = 2



    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):

        super(_DenseBlock, self).__init__()

        for i in range(num_layers):

            layer = _DenseLayer(

                num_input_features + i * growth_rate,

                growth_rate=growth_rate,

                bn_size=bn_size,

                drop_rate=drop_rate,

                memory_efficient=memory_efficient,

            )

            self.add_module('denselayer%d' % (i + 1), layer)



    def forward(self, init_features):

        features = [init_features]

        for name, layer in self.items():

            new_features = layer(features)

            features.append(new_features)

        return torch.cat(features, 1)





class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):

        super(_Transition, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(num_input_features))

        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,

                                          kernel_size=1, stride=1, bias=False))

        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))





class DenseNet(nn.Module):

    r"""Densenet-BC model class, based on

    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:

        growth_rate (int) - how many filters to add each layer (`k` in paper)

        block_config (list of 4 ints) - how many layers in each pooling block

        num_init_features (int) - the number of filters to learn in the first convolution layer

        bn_size (int) - multiplicative factor for number of bottle neck layers

          (i.e. bn_size * k features in the bottleneck layer)

        drop_rate (float) - dropout rate after each dense layer

        num_classes (int) - number of classification classes

        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,

          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_

    """



    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),

                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):



        super(DenseNet, self).__init__()



        # First convolution

        self.features = nn.Sequential(OrderedDict([

            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,

                                padding=3, bias=False)),

            ('norm0', nn.BatchNorm2d(num_init_features)),

            ('relu0', nn.ReLU(inplace=True)),

            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ]))



        # Each denseblock

        num_features = num_init_features

        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(

                num_layers=num_layers,

                num_input_features=num_features,

                bn_size=bn_size,

                growth_rate=growth_rate,

                drop_rate=drop_rate,

                memory_efficient=memory_efficient

            )

            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:

                trans = _Transition(num_input_features=num_features,

                                    num_output_features=num_features // 2)

                self.features.add_module('transition%d' % (i + 1), trans)

                num_features = num_features // 2



        # Final batch norm

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))



        # Linear layer

        self.classifier = nn.Linear(num_features, num_classes)



        # Official init from torch repo.

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.constant_(m.bias, 0)



    def forward(self, x):

        features = self.features(x)

        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = torch.flatten(out, 1)

        out = self.classifier(out)

        return out
def _load_state_dict(model, model_url, progress):

    # '.'s are no longer allowed in module names, but previous _DenseLayer

    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.

    # They are also in the checkpoints in model_urls. This pattern is used

    # to find such keys.

    pattern = re.compile(

        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')



    state_dict = load_state_dict_from_url(model_url, progress=progress)

    for key in list(state_dict.keys()):

        res = pattern.match(key)

        if res:

            new_key = res.group(1) + res.group(2)

            state_dict[new_key] = state_dict[key]

            del state_dict[key]

    model.load_state_dict(state_dict)





def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,

              **kwargs):

    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)

    if pretrained:

        _load_state_dict(model, model_urls[arch], progress)

    return model
def densenet169(pretrained=False, progress=True, **kwargs):

    r"""Densenet-169 model from

    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,

          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_

    """

    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,

                     **kwargs)

## Create base Model for our purpose

model = densenet169(pretrained =True)
## Remove the head from the model because we dont need to do 1000 class classification like imagenet

model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
## make the final pool adaptive

model[0].final_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)))
## Create then head that we need . Copied from the wise people before me .

class Head(torch.nn.Module):

  def __init__(self, in_f, out_f):

    super(Head, self).__init__()

    

    self.f = nn.Flatten()

    self.l = nn.Linear(in_f, 512)

    self.d = nn.Dropout(0.75)

    self.o = nn.Linear(512, out_f)

    self.b1 = nn.BatchNorm1d(in_f)

    self.b2 = nn.BatchNorm1d(512)

    self.r = nn.ReLU()



  def forward(self, x):

    x = self.f(x)

    x = self.b1(x)

    x = self.d(x)



    x = self.l(x)

    x = self.r(x)

    x = self.b2(x)

    x = self.d(x)



    out = self.o(x)

    return out
## Create the final model

class FCN(torch.nn.Module):

  def __init__(self, base, in_f):

    super(FCN, self).__init__()

    self.base = base

    self.h1 = Head(in_f, 1)

  

  def forward(self, x):

    x = self.base(x)

    return self.h1(x)



model = FCN(model, 1664)
## This is our loss

def criterion1(pred1, targets):

  l1 = F.binary_cross_entropy(F.sigmoid(pred1), targets)

  return l1



## This is training pipeline 

def train_model(epoch, optimizer, scheduler=None, history=None):

    model.train()

    total_loss = 0

    

    t = tqdm(train_loader)

    for i, (img_batch, y_batch) in enumerate(t):

        img_batch = img_batch.cuda().float()

        y_batch = y_batch.cuda().float()



        optimizer.zero_grad()



        out = model(img_batch)

        loss = criterion1(out, y_batch)



        total_loss += loss.item()

        t.set_description(f'Epoch {epoch+1}/{n_epochs}, LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(i+1)))



        if history is not None:

          history.loc[epoch + i / len(X), 'train_loss'] = loss.item()#.cpu().numpy()

          history.loc[epoch + i / len(X), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']



        loss.backward()

        optimizer.step()

        if scheduler is not None:

          scheduler.step()



def evaluate_model(epoch, scheduler=None, history=None):

    model.eval()

    loss = 0

    pred = []

    real = []

    with torch.no_grad():

        for img_batch, y_batch in val_loader:

            img_batch = img_batch.cuda().float()

            y_batch = y_batch.cuda().float()



            o1 = model(img_batch)

            l1 = criterion1(o1, y_batch)

            loss += l1

            

            for j in o1:

              pred.append(F.sigmoid(j))

            for i in y_batch:

              real.append(i.data.cpu())

    

    pred = [p.data.cpu().numpy() for p in pred]

    pred2 = pred

    pred = [np.round(p) for p in pred]

    pred = np.array(pred)

    acc = sklearn.metrics.recall_score(real, pred, average='macro')



    real = [r.item() for r in real]

    pred2 = np.array(pred2).clip(0.1, 0.9)

    kaggle = sklearn.metrics.log_loss(real, pred2)



    loss /= len(val_loader)

    

    if history is not None:

        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()

    

    if scheduler is not None:

      scheduler.step(loss)



    print(f'Dev loss: %.4f, Acc: %.6f, Kaggle: %.6f'%(loss,acc,kaggle))

    

    return loss
## Create train , test split

X, val_X, y, val_y = get_random_sampling(paths, y, val_paths, val_y)



print('There are '+str(y.count(1))+' fake train samples')

print('There are '+str(y.count(0))+' real train samples')

print('There are '+str(val_y.count(1))+' fake val samples')

print('There are '+str(val_y.count(0))+' real val samples')
## Needed this library for some out of the box augmentations

import albumentations

from albumentations.core.transforms_interface import DualTransform

from albumentations.augmentations import functional as F1

from albumentations.augmentations.transforms import ShiftScaleRotate, HorizontalFlip, Normalize, RandomBrightnessContrast,RandomBrightness,RandomGridShuffle
## The author details are there along with the function

class GridMask(DualTransform):

    """GridMask augmentation for image classification and object detection.

    

    Author: Qishen Ha

    Email: haqishen@gmail.com

    2020/01/29



    Args:

        num_grid (int): number of grid in a row or column.

        fill_value (int, float, lisf of int, list of float): value for dropped pixels.

        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int

            an angle is picked from (-rotate, rotate). Default: (-90, 90)

        mode (int):

            0 - cropout a quarter of the square of each grid (left top)

            1 - reserve a quarter of the square of each grid (left top)

            2 - cropout 2 quarter of the square of each grid (left top & right bottom)



    Targets:

        image, mask



    Image types:

        uint8, float32



    Reference:

    |  https://arxiv.org/abs/2001.04086

    |  https://github.com/akuxcw/GridMask

    """



    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):

        super(GridMask, self).__init__(always_apply, p)

        if isinstance(num_grid, int):

            num_grid = (num_grid, num_grid)

        if isinstance(rotate, int):

            rotate = (-rotate, rotate)

        self.num_grid = num_grid

        self.fill_value = fill_value

        self.rotate = rotate

        self.mode = mode

        self.masks = None

        self.rand_h_max = []

        self.rand_w_max = []



    def init_masks(self, height, width):

        if self.masks is None:

            self.masks = []

            n_masks = self.num_grid[1] - self.num_grid[0] + 1

            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):

                grid_h = height / n_g

                grid_w = width / n_g

                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)

                for i in range(n_g + 1):

                    for j in range(n_g + 1):

                        this_mask[

                             int(i * grid_h) : int(i * grid_h + grid_h / 2),

                             int(j * grid_w) : int(j * grid_w + grid_w / 2)

                        ] = self.fill_value

                        if self.mode == 2:

                            this_mask[

                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),

                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)

                            ] = self.fill_value

                

                if self.mode == 1:

                    this_mask = 1 - this_mask



                self.masks.append(this_mask)

                self.rand_h_max.append(grid_h)

                self.rand_w_max.append(grid_w)



    def apply(self, image, mask, rand_h, rand_w, angle, **params):

        h, w = image.shape[:2]

        mask = F1.rotate(mask, angle) if self.rotate[1] > 0 else mask

        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask

        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)

        return image



    def get_params_dependent_on_targets(self, params):

        img = params['image']

        height, width = img.shape[:2]

        self.init_masks(height, width)



        mid = np.random.randint(len(self.masks))

        mask = self.masks[mid]

        rand_h = np.random.randint(self.rand_h_max[mid])

        rand_w = np.random.randint(self.rand_w_max[mid])

        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0



        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}



    @property

    def targets_as_params(self):

        return ['image']



    def get_transform_init_args_names(self):

        return ('num_grid', 'fill_value', 'rotate', 'mode')
batch_size= 16

class_sample_count = np.array([len(np.where(y==t)[0]) for t in np.unique(y)])

weight = 1. / class_sample_count

samples_weight = np.array([weight[t] for t in y])

samples_weight = torch.from_numpy(samples_weight)

sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
## Create the transformation we need 

train_transform = albumentations.Compose([

                                          ShiftScaleRotate(p=0.3, scale_limit=0.25, border_mode=1),

                                          HorizontalFlip(p=0.2),

                                          albumentations.RandomCrop(150,150),

                                          #A.RandomBrightness(limit=0.3, always_apply=False, p=0.3),

                                         # albumentations.RandomGridShuffle(grid=(2, 1), always_apply=False, p=0.4),

                                          RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),

                                          albumentations.OneOf([

                                            GridMask(num_grid=(1,3),rotate=15),

                                            GridMask(num_grid=(2,4), mode=0),

                                            GridMask(num_grid=3, mode=2),

                                          ], p=0.5),

                                          Normalize()

])

val_transform = albumentations.Compose([

                                        albumentations.RandomCrop(150,150),  

                                        Normalize()

])



train_dataset = ImageDataset(X, y, transform=train_transform)

val_dataset = ImageDataset(val_X, val_y, transform=val_transform)



train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,sampler= sampler,pin_memory=True)

val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
import gc

gc.collect()
nrow, ncol = 5, 6

fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))

axes = axes.flatten()

for i, ax in enumerate(axes):

    image, label = train_dataset[i]

    image = np.rollaxis(image, 0, 3)

    image = image*std + mean

    image = np.clip(image, 0., 1.)

    ax.imshow(image)

    ax.set_title(f'label: {label}')
import gc



history = pd.DataFrame()

history2 = pd.DataFrame()



torch.cuda.empty_cache()

gc.collect()



best = 1e10

n_epochs = 5

batch_size = batch_size



model = model.cuda()



optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=1e-5)



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, mode='min', factor=0.1, verbose=True, min_lr=1e-5)



for epoch in range(n_epochs):

    torch.cuda.empty_cache()

    gc.collect()



    train_model(epoch, optimizer, scheduler=None, history=None)

    

    loss = evaluate_model(epoch, scheduler=scheduler, history=history2)

    

    if loss < best:

      best = loss

      print(f'Saving best model...')

      torch.save(model.state_dict(), f'model.pth')
history2.plot()