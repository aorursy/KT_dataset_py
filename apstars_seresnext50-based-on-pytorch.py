import pandas as pd

import numpy as np

import torch

import torch.optim as optim

import torch.nn.functional as F

import os

import cv2

import torchvision

import sklearn.metrics



from tqdm import tqdm

from torch.utils.data import Dataset

from albumentations import Compose, ShiftScaleRotate, Resize

from albumentations.pytorch import ToTensorV2



INPUT_PATH = '/kaggle/input/bengaliai-cv19'

INPUT_PATH_TRAIN_IMAGES = '/kaggle/input/bengaliai/256_train/256'
from __future__ import print_function, division, absolute_import

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch.nn.functional as F

import os



# Any results you write to the current directory are saved as output.

import torch

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

from torchvision import transforms,models

from tqdm import tqdm_notebook as tqdm

import math

import torch.utils.model_zoo as model_zoo
# ======================

# Params

BATCH_SIZE = 32

N_WORKERS = 4

N_EPOCHS = 100



# Disable training in kaggle

TRAIN_ENABLED = False
class BengaliImageDataset(Dataset):



    def __init__(self, csv_file, path, labels, transform=None):



        self.data = pd.read_csv(csv_file)

        self.data_dummie_labels = pd.get_dummies(

            self.data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']],

            columns=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']

        )

        self.path = path

        self.labels = labels

        self.transform = transform



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):



        image_name = os.path.join(self.path, self.data.loc[idx, 'image_id'] + '.png')

        img = cv2.imread(image_name)



        if self.transform:

            transformed = self.transform(image=img)

            img = transformed['image']



        if self.labels:

            return {

                'image': img,

                'l_graph': torch.tensor(self.data_dummie_labels.iloc[idx, 0:168]),

                'l_vowel': torch.tensor(self.data_dummie_labels.iloc[idx, 168:179]),

                'l_conso': torch.tensor(self.data_dummie_labels.iloc[idx, 179:186]),

            }

        else:

            return {'image': img}

class SElayer(nn.Module):

    def __init__(self, inplanes, reduction=16):

        super(SElayer,self).__init__()

        self.globalAvgpool = nn.AdaptiveAvgPool2d(1)#Squeeze操作

        self.fc1 = nn.Conv2d(inplanes, inplanes // reduction, kernel_size=1, stride=1)

        self.fc2 = nn.Conv2d(inplanes // reduction, inplanes, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        begin_input = x

        x = self.globalAvgpool(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        

        return x * begin_input

                



class Bottleneck(nn.Module):

    expansion = 4

    

    def __init__(self,inplanes,outplanes,stride=1,cardinality=32, downsample=None):

        super(Bottleneck,self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes*2, kernel_size = 1, bias = False)

        self.bn1 = nn.BatchNorm2d(outplanes*2)

        self.conv2 = nn.Conv2d(outplanes*2, outplanes*2,kernel_size=3,stride=stride,padding=1,

                               bias=False,groups = cardinality)

        self.bn2 = nn.BatchNorm2d(outplanes*2)

        self.conv3 = nn.Conv2d(outplanes*2, outplanes*4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(outplanes*4)

        self.selayer = SElayer(outplanes*4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride

        

    def forward(self,x):

        residual = x

        

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        

        x = self.conv2(x)

        x = self.bn2(x)

        x = self.relu(x)



        x = self.conv3(x)

        x = self.bn3(x)

        

        x = self.selayer(x)

        

        if self.downsample is not None:

            residual = self.downsample(x)

            

        x += residual

        x = self.relu(x)

        

        return x



class SEResNext(nn.Module):

    

    def __init__(self, block, layers, cardinality = 32,num_classes = 1000):

        super(SEResNext, self).__init__()

        self.cardinality = cardinality

        self.inplanes = 64

        

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,

                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        

        

        #参数初始化，待懂

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:

                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()

        

    def _make_layer(self, block, outplanes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != outplanes * block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.inplanes, outplanes * block.expansion,

                          kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(outplanes * block.expansion),

            )



        layers = []

        layers.append(block(self.inplanes, outplanes, self.cardinality, stride, downsample))

        self.inplanes = outplanes * block.expansion

        for i in range(1, blocks):

            layers.append(block(self.inplanes, outplanes, self.cardinality))





        return nn.Sequential(*layers)

        

        def forward(self, x):

            x = self.conv1(x)

            x = self.bn1(x)

            x = self.relu(x)

            x = self.maxpool(x)

    

            x = self.layer1(x)

            x = self.layer2(x)

            x = self.layer3(x)

            x = self.layer4(x)

    

            x = self.avgpool(x)

            x = x.view(x.size(0), -1)

            

            x = self.fc(x)

            

            fc_graph = torch.nn.Linear(x.in_features, 168)

            fc_vowel = torch.nn.Linear(x.in_features, 11)

            fc_conso = torch.nn.Linear(in_features, 7)

            

            return fc_graph, fc_vowel, fc_conso

        

        

def se_resnext50(**kwargs):

    

    model = SEResNext(Bottleneck, [3,4,6,3], **kwargs)

    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform_train = Compose([

    ToTensorV2()

])



train_dataset = BengaliImageDataset(

    csv_file=INPUT_PATH + '/train.csv',

    path=INPUT_PATH_TRAIN_IMAGES,

    transform=transform_train, labels=True

)



data_loader_train = torch.utils.data.DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    num_workers=N_WORKERS,

    shuffle=True

)



model = se_resnext50().to(device)

optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.05)

criterion = nn.CrossEntropyLoss()

batch_size=32
# TRAIN_ENABLED is just for faster committing.

# Feel free to remove it.

if TRAIN_ENABLED:

    for epoch in range(N_EPOCHS):



        print('Epoch {}/{}'.format(epoch, N_EPOCHS - 1))

        print('-' * 10)



        model.train()

        tr_loss = 0



        tk0 = tqdm(data_loader_train, desc="Iteration")



        for step, batch in enumerate(tk0):

            inputs = batch["image"]

            l_graph = batch["l_graph"]

            l_vowel = batch["l_vowel"]

            l_conso = batch["l_conso"]



            inputs = inputs.to(device, dtype=torch.float)

            l_graph = l_graph.to(device, dtype=torch.float)

            l_vowel = l_vowel.to(device, dtype=torch.float)

            l_conso = l_conso.to(device, dtype=torch.float)



            out_graph, out_vowel, out_conso = model(inputs)



            loss_graph = criterion(out_graph, l_graph)

            loss_vowel = criterion(out_vowel, l_vowel)

            loss_conso = criterion(out_conso, l_conso)



            loss = loss_graph + loss_vowel + loss_conso

            loss.backward()



            tr_loss += loss.item()



            optimizer.step()

            optimizer.zero_grad()



        epoch_loss = tr_loss / len(data_loader_train)

        print('Training Loss: {:.4f}'.format(epoch_loss))



torch.save(model.state_dict(), './seresnext_model.pth')