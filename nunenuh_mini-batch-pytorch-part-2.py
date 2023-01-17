%matplotlib inline
import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import pandas as pd
import seaborn as sns
# !ls -al /kaggle/input/iris
p = Path('/kaggle/input/iris/Iris.csv')
df = pd.read_csv(p)
df = df.drop(columns='Id')
df['Species'] = df['Species'].astype('category').cat.codes
feature = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
target = df[['Species']]
feature
class IrisDataset(torch.utils.data.Dataset):
    def __init__(self, path, feature_columns, target_columns, transform=None):
        self.path = Path(path)
        self.dframe = pd.read_csv(self.path)
        self._do_normalizer()
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.transform = transform

    def _do_normalizer(self):
        self.dframe = self.dframe.drop(columns='Id')
        self.dframe['Species'] = self.dframe['Species'].astype('category').cat.codes
        
    def __len__(self):
        return len(self.dframe)
    
    def __getitem__(self, idx):
        feature = self.dframe[self.feature_columns].iloc[idx].values
        target = self.dframe[self.target_columns].iloc[idx].values
        
        if self.transform:
            feature = self.transform(feature)
            target = self.transform(target)

        return feature, target

class NumpyToTensor(object):
    def __call__(self, param):
        return torch.from_numpy(param.astype(np.float32))
p = '/kaggle/input/iris/Iris.csv'
fc = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
tc = ['Species']
tmft = transforms.Compose([NumpyToTensor()])
iris = IrisDataset(path=p,feature_columns=fc, target_columns=tc, transform=tmft )
loader = torch.utils.data.DataLoader(iris, batch_size=16, shuffle=True, num_workers=0)
[1]
[0,1,0]

[2]
[0,0,1]

[0]
[1,0,0]
load_iter = iter(loader)
x,y = load_iter.next()
import torch.nn as nn
linear = nn.Linear(4,1)
linear.weight
# out = linear(x)

# class NumpyToTensor(object):
#     def __call__(self, param):
#         return torch.from_numpy(param.astype(np.float32))
    
# class TensorToNumpy(object):
#     def __call__(self, param):
#         return param.numpy()
    
# a = np.array([1,5,6])
# tmft = transforms.Compose([
#     NumpyToTensor(),
#     TensorToNumpy()
# ])
# x = tmft(a)

# ntt = NumpyToTensor(),
# ttn = TensorToNumpy()
# x = ntt(a)
# x = ttn(x)
# x
from pathlib import Path
path = Path('/kaggle/input/flower_data/flower_data/')
list(path.iterdir())
train_path = path.joinpath('train')
list(train_path.glob('*'))
trainset = torchvision.datasets.ImageFolder(train_path)
trainset.__len__()
image, label = trainset.__getitem__(0)
trainset.class_to_idx
class FandiDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super(FandiDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.dirfiles = sorted(glob.glob(self.path+'/*'))
        self.files =  sorted(glob.glob(self.path+'/*/*.jpg'))
        self.class_to_idx = self._class_to_idx()
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fpath = self.files[idx]
        img = PIL.Image.open(fpath)
        label_key = fpath.split('/')[-2]
        label_idx = self.class_to_idx[label_key]
        
        if self.transform:
            img = self.transform(img)
#         img = torch.from_numpy(np.array(img))
            
        return img, label_idx

    def _class_to_idx(self):
        data = {}
        for idx in range(len(self.dirfiles)):
            label = self.dirfiles[idx].split('/')[-1]
            data.update({label:idx})
        return data

path= '/kaggle/input/flower_data/flower_data/'
train_path = os.path.join(path, 'train')

tmft = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_set = FandiDataset(train_path, transform=tmft)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
class ToufanDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = Path(path)
        self.dirfiles = sorted(list(train_path.glob('*')))
        self.files = sorted(list(train_path.glob('*/*.jpg')))
        self.class_to_idx = self._class_to_idx()
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def _class_to_idx(self):
        data = {}
        for idx in range(len(self.dirfiles)):
            dirname = str(self.dirfiles[idx]).split('/')[-1]
            data.update({dirname:idx})
        return data
    
    def __getitem__(self, idx):
        pf = str(self.files[idx])
        dirname = pf.split('/')[-2]
        label = data[dirname]
        img = PIL.Image.open(pf)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
        
tp = '/kaggle/input/flower_data/flower_data/'
train_dataset = ToufanDataset(tp)
img, label = train_dataset.__getitem__(0)
np.array(img).shape
 
tmft = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
]) 
tp = '/kaggle/input/flower_data/flower_data/'
train_dataset = ToufanDataset(tp, transform=tmft)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
train_iter = iter(train_loader)
img, label = train_iter.next()
plt.imshow(img[0].permute(1,2,0), cmap='gray')
path = Path('/kaggle/input/flower_data/flower_data/')
train_path = path.joinpath('train')
dirfiles = sorted(list(train_path.glob('*')))
files = sorted(list(train_path.glob('*/*.jpg')))

data = {}
for idx in range(len(dirfiles)):
    dirname = str(dirfiles[idx]).split('/')[-1]
    data.update({dirname:idx})

# data['1']
idx = 0
pf = str(files[idx])
dirname = pf.split('/')[-2]
label = data[dirname]
img = PIL.Image.open(pf)

# class HitamPutih(object):
#     def __call__(self, img):
#         return img.convert('L')


# tmft = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
# ])

# x,y = train_set.__getitem__(0)
# x = tmft(x)
# x
# path = '/data/flower_data'
# train_path = os.path.join(path, 'train')
# dirfiles = sorted(glob.glob(train_path+'/*'))
# data = {}
# for idx in range(len(dirfiles)):
#     label = dirfiles[idx].split('/')[-1]
#     data.update({label:idx})

import torch.nn as nn
linear1 = nn.Linear(3,4)
linear2 = nn.Linear(4,1)

inp = torch.FloatTensor([[1,0,1]])
x = linear1(inp)
x = linear2(x)
x
