# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
from torch import nn
import cv2
gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gpu

#delete it
import torch.nn.functional as F
x = torch.arange(24).view(3,2,4)
print(x)
print(x)
F.softmax(x.float(), dim = -1)
img_PATH = '/kaggle/input/adapt-to-faceforensics/img_data'
metadata = pd.read_csv('/kaggle/input/adapt-to-faceforensics/img_data/metadata.csv')
test_val_frac = 0.3
frames_per_video = 17
input_size =224
from torchvision.transforms import Normalize
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)
import matplotlib.pyplot as plt
def make_split(df, frac):
    val_df = df.sample(frac=frac, random_state=666)
    train_df=df.loc[df.index.isin(val_df.index)]
    val_df.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    return train_df, val_df


def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size
    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized

def make_square_img(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
def load_image(filename, label):
    
    X = list()
    for i in range(frames_per_video):
        p = os.path.join(img_PATH, label, filename[:-4]+'_img_'+str(i)+'.jpg')
        img = cv2.imread(p)
        if not os.path.isfile(p): #some videos may not generate frams_per_video much images
            continue 
        img = isotropically_resize_image(img, input_size)
        img = make_square_img(img)
        img = torch.tensor(img).float()

        img = img.permute(2,0,1)
        img = normalize_transform(img/255)
        X.append(img)
    X = torch.stack(X, axis=0)
    while len(X) != frames_per_video:
        X = torch.cat((X, X[-1].reshape((1,)+X[-1].shape)))
    y = 0 if label=='Pristine' else 1
    y = torch.tensor([y]*len(X))
    return X,  y
(1,2)+(3,)
from torch.utils.data import Dataset
class ImgDataset(Dataset):
    def __init__(self,df):
        self.df = df
    def __getitem__(self, index):
        filename = self.df['filename'][index]
        label = self.df['category'][index]
        return load_image(filename, label)
    def __len__(self):
        return len(self.df)
train_df, val_df = make_split(metadata, frac=test_val_frac)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
train_data = ImgDataset(train_df)
val_data = ImgDataset(val_df)
train_iter = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
val_iter = DataLoader(val_data, batch_size=32, shuffle=True, pin_memory=True)


            
import torchvision.models as models
checkpoint = torch.load("../input/pretrained-pytorch/resnext50_32x4d-7cdf4587.pth")
class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3], 
                                        groups=32, 
                                        width_per_group=4)


        # Override the existing FC layer with a new one.
        self.load_state_dict(checkpoint)
        self.fc = nn.Linear(2048, 1)
def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name

net = MyResNeXt().to(gpu)
freeze_until(net, "layer4.0.conv1.weight")


import torch.nn.functional as F
from tqdm.notebook import tqdm
def evaluate(net, data_loader, device, silent=False):
    net.train(False)
    num_examples = 0.
    bce_loss = 0. # binary cross entropy loss
    accurate_examples = 0.
    
    with tqdm(total=len(data_loader), desc="Evaluation", leave=False, disable=silent) as pbar:
        for batch_index, data in enumerate(data_loader):
            with torch.no_grad():
                X = data[0].to(device)
                y=data[1].float().mean(1).to(device)
                batch_size = X.shape[0]
                y_hat = list()
                X = X.permute(1,0,2,3,4)
                for frame_X in X:
                    y_hat.append(net(frame_X).view(-1))
                y_hat = torch.stack(y_hat).to(device)
                y_hat = y_hat.mean(0)
                bce_loss += F.binary_cross_entropy_with_logits(y_hat, y).item() * batch_size
                accurate_examples += ((torch.sigmoid(y_hat)>0.5)==y).sum()
            num_examples +=batch_size
            pbar.update()
    bce_loss /= num_examples
    accuracy = accurate_examples/num_examples
    
    if silent:
        return bce_loss
    else:
        print("BCE: %.4f" % (bce_loss))
        print("accuracy: %.4f" % (accuracy))
    return bce_loss, accuracy
    
evaluate(net, val_iter, gpu)         
            
                
import time
history = { "train_bce": [], "val_bce": [] }
def train(epochs, net, optimizer, train_loader, val_loader, device, silent=False):
    global history
    with tqdm(total=len(train_loader), leave=False) as bar:
        for epoch in range(epochs):
            bar.reset()
            bar.set_description("Epoch %d"%(epoch))
            bce_loss = 0.
            num_examples = 0.
            start_time = time.time()
            accurate_exampls = 0
            
            for batch_index, data in enumerate(train_loader):
                X = data[0].to(device)
                y=data[1].float().to(device)
                X = X.view((-1,)+X.shape[2:])
                y = y.view((-1,))
                batch_size = len(y)
                optimizer.zero_grad()
                
                y_hat = net(X).squeeze()
                assert y.shape == y_hat.shape
                #print(y_hat)
                accurate_exampls+= ((torch.sigmoid(y_hat)>0.5)==y).sum()
                loss = F.binary_cross_entropy_with_logits(y_hat, y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    num_examples += batch_size
                    bce_loss+=loss.item()*batch_size
                bar.update()
            
            end_time = time.time()
            print("Epoch: %3d, train BCE: %.4f, accuracy: %.4f, speed: %.4f examples/s" % \
                  (epoch,  bce_loss/num_examples,accurate_exampls/num_examples, num_examples/(end_time-start_time)))
            print("batch_size: %d"%(batch_size))
            evaluate(net, val_loader, device=device)
                
            print("")
            
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.)
train(5, net, optimizer, train_iter, val_iter, gpu, silent=False )
torch.save(net.state_dict(), "checkpoint.pth")
os.listdir(".")