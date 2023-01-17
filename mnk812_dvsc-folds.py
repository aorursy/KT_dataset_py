import pandas as pd

#import config

import os

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm
!pip install pretrainedmodels
from zipfile import ZipFile



test_file_name = "../input/dogs-vs-cats-redux-kernels-edition/test.zip"

with ZipFile(test_file_name, 'r') as zip: 

    print('Extracting all the files now...') 

    zip.extractall('../output/kaggle/working/') 

    print('Done!') 
train_file_name = "../input/dogs-vs-cats-redux-kernels-edition/train.zip"



with ZipFile(train_file_name, 'r') as zip: 

    print('Extracting all the files now...') 

    zip.extractall('../output/kaggle/working/') 

    print('Done!') 
class config:

    TRAIN_DIR = '../output/kaggle/working/train/'

    TEST_DIR = '../output/kaggle/working//test/'

    EPOCHS = 3

    IMG_HEIGHT = 224

    IMG_WIDTH = 224

    TRAIN_BATCH_SIZE = 32

    TEST_BATCH_SIZE = 16

    BASE_MODEL = 'resnet34'
train_images = [i for i in os.listdir(config.TRAIN_DIR)]

train = pd.DataFrame({'id':train_images})

train['label'] = train['id'].apply(lambda x: str(x).split('.')[0])

labels = {'cat':0, 'dog':1}

train['label'] = train['label'].map(labels)

train['kfold'] = -1

train = train.sample(frac=1).reset_index(drop=True)

kf = StratifiedKFold(n_splits=5)

for fold, (trn_, val_) in enumerate(kf.split(X=train, y=train.label.values)):

    print(len(trn_), len(val_))

    train.loc[val_, 'kfold'] = fold



train.head()
import pandas as pd

import cv2

import albumentations

import numpy as np

import torch

#import config





class DogCatDataset:

    def __init__(self, ids, labels, transform=None):

        super(DogCatDataset, self).__init__()

        self.ids = ids

        self.labels = labels

        self.transform = transform



    def __len__(self):

        return len(self.ids)



    def __getitem__(self, item):

        img = cv2.imread(config.TRAIN_DIR+self.ids[item])

        img = cv2.resize(img,(config.IMG_HEIGHT,config.IMG_WIDTH), interpolation = cv2.INTER_AREA)

        

        if self.transform is not None:

            res = self.transform(image=img)

            img = res['image'].astype(np.float32)

        

        img = img.transpose((2,0,1))

        label = self.labels[item]





        return {

            'image': torch.tensor(img, dtype=torch.float),

            'label': torch.tensor(label, dtype=torch.long),

        }
import torch.nn as nn

import pretrainedmodels

from torch.nn import functional as F







class ResNet34(nn.Module):

    def __init__(self, pretrained=True):

        super(ResNet34, self).__init__()

        if pretrained is True:

            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')

        else:

            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)



        self.l0 = nn.Linear(512, 2)

        



    def forward(self, x):

        bs, _, _, _ = x.shape

        x = self.model.features(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        l0 = self.l0(x)

        

        return l0
MODEL_DISPATCHER = {

    'resnet34':ResNet34

}
def loss_fn(outputs, targets):

    loss = nn.CrossEntropyLoss()(outputs, targets)

    return loss
def train_epoch(dataset, data_loader, model, optimizer, device):

    model.train()



    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):

        image = d['image']

        label = d['label']



        image = image.to(device, dtype=torch.float)

        label = label.to(device, dtype=torch.long)

        optimizer.zero_grad()

        output = model(image)

        loss = loss_fn(output, label)



        loss.backward()

        optimizer.step()
def evaluate_epoch(dataset, data_loader, model, device):

    model.eval()

    final_loss = 0

    counter = 0

    with torch.no_grad():

        for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):

            counter = counter + 1

            image = d['image']

            label = d['label']



            image = image.to(device, dtype=torch.float)

            label = label.to(device, dtype=torch.long)



            outputs = model(image)

            loss = loss_fn(outputs, label)

            final_loss += loss.detach().cpu().numpy()

        return final_loss / counter
def run(fold):

    df_train = train[train.kfold != fold].reset_index(drop=True)

    df_valid = train[train.kfold == fold].reset_index(drop=True)



    train_dataset = DogCatDataset(

        ids = df_train.id.values,

        labels = df_train.label.values

    )



    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=config.TRAIN_BATCH_SIZE,

        num_workers=4

    )



    valid_dataset = DogCatDataset(

        ids = df_valid.id.values,

        labels = df_valid.label.values

    )



    valid_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=config.TRAIN_BATCH_SIZE,

        num_workers=4

    )



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MODEL_DISPATCHER[config.BASE_MODEL](pretrained=True)

    model.to(device)



    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.3, verbose=True)



    if torch.cuda.device_count() > 1:

        model = nn.DataParallel(model)



    best_val_score = 100

    for epoch in range(config.EPOCHS):

        train_epoch(train_dataset, train_loader, model, optimizer, device)

        val_score = evaluate_epoch(valid_dataset, valid_loader, model, device)

        print(val_score)

        scheduler.step(val_score)

        if val_score < best_val_score:

            print('save model!')

            best_val_score = val_score

            torch.save(model.state_dict(), f"{config.BASE_MODEL}_{fold}.bin")
run(0)
run(1)
run(2)
run(3)
run(4)