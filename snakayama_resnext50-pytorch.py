# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_submission = pd.read_csv('../input/sample_submission.csv')
df_train['diagnosis'].hist()
df_train.head()
import torch

import torch.nn.functional as F

import argparse

import cv2

import numpy as np

from glob import glob

import copy

from tqdm import tqdm

import matplotlib.pyplot as plt
num_classes = 5

img_height, img_width = 96, 96

channel = 3

GPU = True

torch.manual_seed(0)
class ResNeXtBlock(torch.nn.Module):

    def __init__(self, in_f, f_1, out_f, stride=1, cardinality=32):

        super(ResNeXtBlock, self).__init__()



        self.stride = stride

        self.fit_dim = False

        

        self.block = torch.nn.Sequential(

            torch.nn.Conv2d(in_f, f_1, kernel_size=1, padding=0, stride=stride),

            torch.nn.BatchNorm2d(f_1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(f_1, f_1, kernel_size=3, padding=1, stride=1, groups=cardinality),

            torch.nn.BatchNorm2d(f_1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(f_1, out_f, kernel_size=1, padding=0, stride=1),

            torch.nn.BatchNorm2d(out_f),

            torch.nn.ReLU(),

        )



        if in_f != out_f:

            self.fit_conv = torch.nn.Conv2d(in_f, out_f, kernel_size=1, padding=0, stride=1)

            self.fit_dim = True

            

    def forward(self, x):

        res_x = self.block(x)



        if self.fit_dim:

            x = self.fit_conv(x)



        if self.stride == 2:

            x = F.max_pool2d(x, 2, stride=2)



        x = torch.add(res_x, x)

        x = F.relu(x)

        

        return x
class ResNeXt50(torch.nn.Module):

    def __init__(self):

        super(ResNeXt50, self).__init__()



        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)

        self.bn1 = torch.nn.BatchNorm2d(64)

        

        

        self.block2_1 = ResNeXtBlock(64, 64, 256)

        self.block2_2 = ResNeXtBlock(256, 64, 256)

        self.block2_3 = ResNeXtBlock(256, 64, 256)



        self.block3_1 = ResNeXtBlock(256, 128, 512, stride=2)

        self.block3_2 = ResNeXtBlock(512, 128, 512)

        self.block3_3 = ResNeXtBlock(512, 128, 512)

        self.block3_4 = ResNeXtBlock(512, 128, 512)



        self.block4_1 = ResNeXtBlock(512, 256, 1024, stride=2)

        self.block4_2 = ResNeXtBlock(1024, 256, 1024)

        self.block4_3 = ResNeXtBlock(1024, 256, 1024)

        self.block4_4 = ResNeXtBlock(1024, 256, 1024)

        self.block4_5 = ResNeXtBlock(1024, 256, 1024)

        self.block4_6 = ResNeXtBlock(1024, 256, 1024)



        self.block5_1 = ResNeXtBlock(1024, 512, 2048, stride=2)

        self.block5_2 = ResNeXtBlock(2048, 512, 2048)

        self.block5_3 = ResNeXtBlock(2048, 512, 2048)

        

        self.linear = torch.nn.Linear(2048, num_classes)

        

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = F.relu(x)

        x = F.max_pool2d(x, 3, padding=1, stride=2)



        x = self.block2_1(x)

        x = self.block2_2(x)

        x = self.block2_3(x)



        x = self.block3_1(x)

        x = self.block3_2(x)

        x = self.block3_3(x)

        x = self.block3_4(x)



        x = self.block4_1(x)

        x = self.block4_2(x)

        x = self.block4_3(x)

        x = self.block4_4(x)

        x = self.block4_5(x)

        x = self.block4_6(x)



        x = self.block5_1(x)

        x = self.block5_2(x)

        x = self.block5_3(x)



        x = F.avg_pool2d(x, [img_height//32, img_width//32], padding=0, stride=1)

        x = x.view(list(x.size())[0], -1)

        x = self.linear(x)

        x = F.softmax(x, dim=1)

        

        return x
# get train data

def data_load(path, hf=False, vf=False, rot=False):

    xs = []

    ts = []

    paths = []

    

    for dir_path in tqdm(glob(path + '/*')):

        x = cv2.imread(dir_path)

        x = cv2.resize(x, (img_width, img_height)).astype(np.float32)

        x /= 255.

        x = x[..., ::-1]

        name = dir_path.split("/")[3].split(".png")[0]

        xs.append(x)

        #print(name)

        #print(df_train[df_train['id_code']==name]['diagnosis'].values[0])

        ichi = dir_path.split("/")[2]

        if 'train_images' == ichi:

            t = df_train[df_train['id_code']==name]['diagnosis'].values[0]

        else:

            t = 0

        ts.append(t)



        paths.append(dir_path)

        

    xs = np.array(xs, dtype=np.float32)

    ts = np.array(ts, dtype=np.int)

    

    xs = xs.transpose(0,3,1,2)



    return xs, ts, paths
# train

def train():

    # GPU

    device = torch.device("cuda" if GPU else "cpu")



    # model

    model = ResNeXt50().to(device)

    opt = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    model.train()



    xs, ts, paths = data_load('../input/train_images/')



    # training

    mb = 128

    mbi = 0

    train_ind = np.arange(len(xs))

    np.random.seed(0)

    np.random.shuffle(train_ind)



    loss_fn = torch.nn.CrossEntropyLoss()

    

    for i in range(5000):

        if mbi + mb > len(xs):

            mb_ind = copy.copy(train_ind)[mbi:]

            np.random.shuffle(train_ind)

            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))

        else:

            mb_ind = train_ind[mbi: mbi+mb]

            mbi += mb



        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)

        t = torch.tensor(ts[mb_ind], dtype=torch.long).to(device)



        opt.zero_grad()

        y = model(x)

        #y = F.log_softmax(y, dim=1)

        loss = loss_fn(y, t)

        

        loss.backward()

        opt.step()

    

        pred = y.argmax(dim=1, keepdim=True)

        acc = pred.eq(t.view_as(pred)).sum().item() / mb



        if (i + 1) % 50 == 0:

            print("iter >>", i+1, ', loss >>', loss.item(), ', accuracy >>', acc)



    torch.save(model.state_dict(), 'cnn.pt')
# test

def test():

    device = torch.device("cuda" if GPU else "cpu")

    model = ResNeXt50().to(device)

    model.eval()

    model.load_state_dict(torch.load('cnn.pt'))



    xs, ts, paths = data_load('../input/test_images/')

    

    pred_list =[]



    for i in range(len(paths)):

        x = xs[i]

        t = ts[i]

        path = paths[i]

        

        x = np.expand_dims(x, axis=0)

        x = torch.tensor(x, dtype=torch.float).to(device)

        

        pred = model(x)

        pred = F.softmax(pred, dim=1).detach().cpu().numpy()[0]

    

        #print("in {}, predicted probabilities >> {}".format(path, pred))

        pred_list.append(np.argmax(pred))

    

    return pred_list
train()
pre = test()
df_submission['diagnosis'] = pre
df_submission['diagnosis'].hist(bins=5)
df_submission.to_csv('submission.csv',index=False)