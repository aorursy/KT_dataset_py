import os

GPU_id = 0

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
import warnings

warnings.filterwarnings("ignore")



from fastai.train import Learner

from fastai.train import DataBunch

from fastai.metrics import accuracy as fastai_accuracy

from fastai.callbacks import SaveModelCallback

from fastai.basic_data import DatasetType



import pandas as pd

import time

import math

from tqdm import tqdm



import torch

import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

from torch import nn,optim

import torch.nn.functional as F



%matplotlib inline
USE_GPU = torch.cuda.is_available()

if USE_GPU:

    print('Use GPU')

else:

    print('Use CPU')
def show_mnist_batch(imgs,labels):

    """Show image for a batch of samples."""



    grid = utils.make_grid(imgs)

    plt.imshow(grid.numpy().transpose((1, 2, 0)))
def cross_entropy(y,yp):

    # y is the ground truch

    # yp is the prediction

    yp[yp>0.99999] = 0.99999

    yp[yp<1e-5] = 1e-5

    return np.mean(-np.log(yp[range(yp.shape[0]),y.astype(int)]))



def accuracy(y,yp):

    return (y==np.argmax(yp,axis=1)).mean()



def softmax(score):

    score = np.asarray(score, dtype=float)

    score = np.exp(score-np.max(score))

    score = score/(np.sum(score, axis=1).reshape([score.shape[0],1]))#[:,np.newaxis]

    return score
class MnistDataset(Dataset):

    """Face Landmarks dataset."""



    def __init__(self, df, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.df = df

        self.transform = transform

        if 'label' in df.columns:

            self.labels = df['label'].values

            self.images = df.drop('label',axis=1).values

        else:

            self.labels = np.zeros(df.shape[0])

            self.images = df.values

        self.images = (self.images/255.0).astype(np.float32).reshape(df.shape[0],28,28)

        

    

    def head(self):

        return self.df.head()



    def __len__(self):

        return self.df.shape[0]



    def __getitem__(self, idx):

        label = np.array(self.labels[idx])

        image = self.images[idx]

        sample = {'image': image, 'label': label}



        if self.transform:

            sample = self.transform(sample)



        return sample['image'],sample['label']
class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""



    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        # torch image: [C, H, W]

        return {'image': torch.from_numpy(image).unsqueeze(0),

                'label': torch.from_numpy(label)}
class Logistic_Model(nn.Module):

    def __init__(self,num_fea,num_class):

        super().__init__()

        #nn.Linear(input_dim, output_dim)

        self.lin = nn.Linear(num_fea,num_class)



    def forward(self, xb):

        B = xb.size()[0]

        if len(xb.size())>2:

            xb = xb.view(B,-1) # 4D tensor of B,C,H,W -> 2D tensor B,CxHxW

        return self.lin(xb)
class SimpleCNN(torch.nn.Module):

    

    #Our batch shape for input x is (3, 32, 32)

    

    def __init__(self,h,w,c,num_class):

        super(SimpleCNN, self).__init__()

        

        #Input channels = 3, output channels = 18

        self.h = h

        self.w = w

        self.c = c

        self.num_class = num_class

        

        self.conv1 = torch.nn.Conv2d(c, 18, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        

        #4608 input features, 64 output features (see sizing flow below)

        self.fc1 = torch.nn.Linear(18 * h//2 * w//2, 64)

        

        #64 input features, 10 output features for our 10 defined classes

        self.fc2 = torch.nn.Linear(64,num_class)

        

    def forward(self, x):

        

        #Computes the activation of the first convolution

        #Size changes from (3, 32, 32) to (18, 32, 32)

        x = F.relu(self.conv1(x))

        

        #Size changes from (18, 32, 32) to (18, 16, 16)

        x = self.pool(x)

        

        #Reshape data to input to the input layer of the neural net

        #Size changes from (18, 16, 16) to (1, 4608)

        #Recall that the -1 infers this dimension from the other given dimension

        x = x.view(-1, 18 * self.w//2 * self.h//2)

        

        #Computes the activation of the first fully connected layer

        #Size changes from (1, 4608) to (1, 64)

        x = F.relu(self.fc1(x))

        

        #Computes the second fully connected layer (activation applied later)

        #Size changes from (1, 64) to (1, 10)

        x = self.fc2(x)

        return(x)
%%time

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print(train_df.shape, test_df.shape)
train_df.head()
%%time



val_pct = 0.2 # use 20% train data as local validation

is_valid = np.random.rand(train_df.shape[0])<val_pct

train_df, valid_df = train_df.loc[~is_valid], train_df.loc[is_valid]

print(train_df.shape, valid_df.shape)
%%time

train_dataset = MnistDataset(df=train_df,

                            transform=transforms.Compose([

                                               ToTensor()

                                           ]))

valid_dataset = MnistDataset(df=valid_df,

                            transform=transforms.Compose([

                                               ToTensor()

                                           ]))

test_dataset = MnistDataset(df=test_df,

                            transform=transforms.Compose([

                                               ToTensor()

                                           ]))
fig = plt.figure(figsize=(20,8))



for i in range(len(train_dataset)):

    img,label = train_dataset[i]



    print(i, img.shape)



    ax = plt.subplot(1, 4, i + 1)

    plt.tight_layout()

    ax.set_title('Sample #{} Label {}'.format(i,label), fontsize=30)

    ax.axis('off')

    plt.imshow(img.numpy()[0],cmap='gray')



    if i == 3:

        plt.show()

        break
%%time



batch_size = 128

cpu_workers = 8



train_dataloader = DataLoader(train_dataset, batch_size=batch_size,

                        shuffle=True, num_workers=cpu_workers,

                        drop_last=True)



valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,

                        shuffle=False, num_workers=cpu_workers,

                        drop_last=False)



test_dataloader = DataLoader(test_dataset, batch_size=batch_size,

                        shuffle=False, num_workers=cpu_workers,

                        drop_last=False)
databunch = DataBunch(train_dl=train_dataloader, 

                      valid_dl=valid_dataloader, 

                      test_dl=test_dataloader,

                     )
model = SimpleCNN(h=28,w=28,c=1,num_class=10)

learn = Learner(databunch, model, loss_func=F.cross_entropy, metrics=fastai_accuracy)
%%time

learn.lr_find()
learn.recorder.plot()
%%time

learn.fit_one_cycle(50,max_lr=slice(0.01),

        callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy',name='mnist')])
%%time

yp,_ = learn.get_preds()

yp = yp.numpy()
%%time

acc = accuracy(valid_df.label.values,yp)

ce = cross_entropy(valid_df.label.values,yp)

print('Valid ACC: %.4f Cross Entropy:%4f'%(acc,ce))
%%time

yps,_ = learn.get_preds(DatasetType.Test)

yps = yps.numpy()
sub = pd.DataFrame()

sub['ImageId'] = np.arange(yps.shape[0])+1

sub['Label'] = np.argmax(yps,axis=1)

sub.head()
from datetime import datetime

clock = "{}".format(datetime.now()).replace(' ','-').replace(':','-').split('.')[0]

out = 'fastai_%s_acc_%.4f_ce_%.4f.csv'%(clock,acc,ce)

print(out)

sub.to_csv(out,index=False)