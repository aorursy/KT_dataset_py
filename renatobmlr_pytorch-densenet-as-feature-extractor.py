import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import glob
from PIL import Image

import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim

from sklearn.model_selection import train_test_split

print(os.listdir("../input"))
random.seed( 42 )
use_gpu = torch.cuda.device_count() > 0
print("{} GPU's available:".format(torch.cuda.device_count()) )
base_dir = '../input'
train_image_dir = os.path.join(base_dir, 'train')
test_image_dir = os.path.join(base_dir, 'test')
df_train = pd.read_csv(base_dir + '/train.csv')
df_test = pd.DataFrame()
test_image_dir
df_train['path'] = df_train['Id'].map(lambda x: os.path.join(train_image_dir, '{}_green.png'.format(x)))
df_train['target_list'] = df_train['Target'].map(lambda x: [int(a) for a in x.split(' ')])

df_test['path'] = glob.glob(os.path.join(test_image_dir, '*.png'))
df_train.head()
X = df_train['path'].values
y = df_train['target_list'].values

X_test = df_test['path'].values
class CellsDataset(Dataset):

    def __init__(self, X, y=None, transforms=None, nb_organelle=28):
        
        self.nb_organelle = nb_organelle
        self.transform = transforms 
        self.X = X
        self.y = y
            
    def open_rgby(self, path2data): #a function that reads RGBY image
        
        Id = path2data.split('/')[-1].split('_')[0]
        basedir = '/'.join(path2data.split('/')[:-1])
        
        images = np.zeros(shape=(512,512,3))
        colors = ['red','green','blue']
        for i, c in enumerate(colors):
            images[:,:,i] = np.asarray(Image.open(basedir + '/' + Id + '_' + c + ".png"))
        
            yellow_ch = np.asarray(Image.open(basedir + '/' + Id + '_yellow.png'))
            images[:,:,0] += (yellow_ch/2).astype(np.uint8) 
            images[:,:,1] += (yellow_ch/2).astype(np.uint8)

        
        return images.astype(np.uint8)
    
    def __getitem__(self, index):
        
        path2img = self.X[index]
        image = self.open_rgby(path2img)

        if self.y is None:
            labels =np.zeros(self.nb_organelle,dtype=np.int)
        else:
            label = np.eye(self.nb_organelle,dtype=np.float)[self.y[index]].sum(axis=0)
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.X)
class AdjustGamma(object):
    def __call__(self, img):
        return transforms.functional.adjust_gamma(img, 0.8, gain=1)
class AdjustContrast(object):
    def __call__(self, img):
        return transforms.functional.adjust_contrast(img, 2)
class AdjustBrightness(object):
    def __call__(self, img):
        return transforms.functional.adjust_brightness(img, 2)
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std  = np.array([0.229, 0.224, 0.225])

def denormalize(image, mean=imagenet_mean, std=imagenet_std):
    inp = image.transpose((1, 2, 0))
    img = std * inp + mean
    return img
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(), # because the input dtype is numpy.ndarray
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        AdjustGamma(),
        AdjustContrast(),
        ##AdjustBrightness(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]),
}
X_train, X_valid, y_train, y_valid = train_test_split(
     X, y, test_size=0.2, random_state=42)
dsets = {
    'train': CellsDataset(X_train, y_train, transforms=data_transforms['train']),
    'valid': CellsDataset(X_valid, y_valid, transforms=data_transforms['test']),
    'test':  CellsDataset(X_test, None,  transforms=data_transforms['test']),
}
batch_size = 32
random_seed = 3
valid_size = 0.2
shuffle = True
def create_dataLoader(dsets, batch_size, shuffle=False, pin_memory=False):
    
    dset_loaders = {} 
    for key in dsets.keys():
        if key == 'test':
            dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, pin_memory=pin_memory, shuffle=False)
        else:
            dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    return dset_loaders
dset_loaders = create_dataLoader(dsets, batch_size, shuffle, pin_memory=False)
dset_loaders.keys()
def plot_organelles(dset_loaders, is_train = True, preds_test = [], preds_train = []):
    
    X, y = next(iter(dset_loaders))
    X, y = X.numpy(), y.numpy()
    
    plt.figure(figsize=(20,10))
    for i in range(0, 4):
        plt.subplot(1,4,i+1)
        rand_img = random.randrange(0, X.shape[0])
        img = denormalize(X[rand_img,:,:,:])
        img = np.clip(img, 0, 1.0)    
        plt.imshow(img)
        plt.axis('off')
image, label = next(iter(dset_loaders['train']))
print(image.size(), label.size())
plot_organelles(dset_loaders['train'])
class MyDenseNetConv(torch.nn.Module):
    def __init__(self, fixed_extractor = True):
        super(MyDenseNetConv,self).__init__()
        original_model = torchvision.models.densenet161(pretrained=True)
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])
        
        if fixed_extractor:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x

class MyDenseNetDens(torch.nn.Module):
    def __init__(self, nb_out=28):
        super().__init__()
        self.dens1 = torch.nn.Linear(in_features=2208, out_features=512)
        self.dens2 = torch.nn.Linear(in_features=512, out_features=128)
        self.dens3 = torch.nn.Linear(in_features=128, out_features=nb_out)
        
    def forward(self, x):
        x = self.dens1(x)
        x = torch.nn.functional.selu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.dens2(x)
        x = torch.nn.functional.selu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.dens3(x)
        return x

class MyDenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mrnc = MyDenseNetConv()
        self.mrnd = MyDenseNetDens()
    def forward(self, x):
        x = self.mrnc(x)
        x = self.mrnd(x)
        return x 
model = MyDenseNet()
if use_gpu:
    print("Using all GPU's ")
    model = torch.nn.DataParallel(model)
    model.cuda()
    convnet = model.module.mrnc
else:
    convnet = model.mrnc
    print("Using CPU's")
def predict(dset_loaders, model,use_gpu=False):
    
    predictions = []
    labels_lst = []

    ii_n = len(dset_loaders)
    start_time = time.time()

    for i, (inputs, labels) in enumerate(dset_loaders):
                   
        if use_gpu:
          inputs = inputs.cuda()
          labels = labels.cuda()

        inputs = Variable(inputs)
        labels = Variable(labels)

        predictions.append(model(inputs).data)
        labels_lst.append(labels)
        
        print('\rpredict: {}/{}'.format(i, ii_n - 1), end='')
    print(' ok')
    print('Execution time {0:.2f} s'.format(round(time.time()- start_time), 2))
    if len(predictions) > 0:
        return {'pred': torch.cat(predictions, 0), 'true': torch.cat(labels_lst, 0) }
#extract features from images
#convOutput_train = predict(dset_loaders['train'], convnet,use_gpu=use_gpu)
convOutput_valid = predict(dset_loaders['valid'], convnet,use_gpu=use_gpu)
#convOutput_test = predict(dset_loaders['test'], convnet,use_gpu=use_gpu)
#print(convOutput_train['true'].size(), convOutput_train['pred'].size())
print(convOutput_valid['true'].size(), convOutput_valid['pred'].size())
#print(convOutput_test['true'].size(), convOutput_test['pred'].size())
print(convOutput_valid['true'].type(), convOutput_valid['pred'].type())
model_name = 'MyDenseNet'
sav_feats= {
    #'train': (convOutput_train['pred'], convOutput_train['true'], model_name),
    'valid': (convOutput_valid['pred'], convOutput_valid['true'], model_name),
    #'test': (convOutput_test['pred'], convOutput_test['true'], model_name)
}
def save_prediction(path2data,convOutput):
    
    for key in convOutput.keys():
        if convOutput[key][0].is_cuda: 
            data ={'true':convOutput[key][0].cpu().numpy(),
                   'pred':convOutput[key][1].cpu().numpy()}
        else:
            data ={'true':convOutput[key][0].numpy(),
                   'pred':convOutput[key][1].numpy()}
        if not os.path.exists(path2data + key):
            os.makedirs(path2data + key)
        
        print('\nSaving '+convOutput[key][2]+' '+ key) 
        np.savez(path2data+key+"/"+convOutput[key][2]+".npz",**data)
        print('Saved in:'+path2data+key+"/"+convOutput[key][2]+".npz")
save_prediction('../results/', sav_feats)
!cd ../results/ && ls .
