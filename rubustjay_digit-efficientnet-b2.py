import sys

sys.path.insert(0, '/kaggle/input/efficientnet/')
import numpy as np

import pandas as pd

import os

from torch.utils.data import Dataset, DataLoader

from sklearn import model_selection

import random

import torch

from efficientnet_pytorch import EfficientNet

import torch.nn as nn

import datetime

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

import cv2



data_dir = '/kaggle/input/digit-recognizer'





device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Fitter prepared. Device is {device}')



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(42)



train = pd.read_csv(f'{data_dir}/train.csv')

test_data = pd.read_csv(f'{data_dir}/test.csv').values

labels = train['label'].values

data = train.drop(labels=['label'], axis=1).values



X_train, X_valid, y_train, y_valid = model_selection.train_test_split(data, labels, test_size=0.2)



def get_train_transforms():

    return A.Compose(

        [

            A.Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.5),

            A.Cutout(num_holes=8, max_h_size=2, max_w_size=2, fill_value=0, p=0.5),

            A.Cutout(num_holes=8, max_h_size=1, max_w_size=1, fill_value=1, p=0.5),

            A.Resize(32, 32, p=1.),

            ToTensorV2(p=1.0),

        ],

        p=1.0)



def get_valid_transforms():

    return A.Compose(

        [

            A.Resize(32, 32, p=1.),

            ToTensorV2(p=1.0),

        ],

        p=1.0

    )



class DigitDataset(Dataset):

    def __init__(self, X, y, transforms=None):

        super(Dataset, self).__init__()

        self.X = X.reshape(-1, 28, 28).astype(np.float32)

        self.y = y

        self.transforms = transforms



    def __getitem__(self, index: int):

        image, target = self.X[index], self.y[index]

        image = np.stack([image] * 3, axis=-1)

        image /= 255.

        if self.transforms:

            image = self.transforms(image=image)['image']

        return image, torch.tensor(target, dtype=torch.long)



    def __len__(self):

        return self.y.shape[0]



class DigitTestDataset(Dataset):

    def __init__(self, X, transforms=None):

        super(Dataset, self).__init__()

        self.X = X.reshape(-1, 28, 28).astype(np.float32)

        self.transforms = transforms



    def __getitem__(self, index: int):

        image = self.X[index]

        image = np.stack([image] * 3, axis=-1)

        image /= 255.

        if self.transforms:

            image = self.transforms(image=image)['image']

        return image



    def __len__(self):

        return self.X.shape[0]



class LossMeter:

    def __init__(self):

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count



class AccMeter:

    def __init__(self):

        self.true_count = 0

        self.all_count = 0

        self.avg = 0



    def update(self, y_true, y_pred):

        y_true = y_true.cpu().numpy().astype(int)

        y_pred = y_pred.cpu().numpy().argmax(axis=1).astype(int)

        self.true_count += (y_true == y_pred).sum()

        self.all_count += y_true.shape[0]

        self.avg = self.true_count / self.all_count



def train(model, train_loader, valid_loader):

    n_epochs = 10

    lr = 0.001

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    scheduler_params = dict(

        mode='min',

        factor=0.5,

        patience=1,

        verbose=False,

        threshold=0.0001,

        threshold_mode='abs',

        cooldown=0,

        min_lr=1e-8,

        eps=1e-08

    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = scheduler(optimizer, **scheduler_params)

    criterion = criterion.to(device)

    model.train()

    for epoch in range(n_epochs):

        current_lr = optimizer.param_groups[0]['lr']

        print(f'\n{datetime.datetime.utcnow().isoformat()}\nLR: {current_lr}')



        summary_loss = LossMeter()

        final_scores = AccMeter()

        for iter, (images, targets) in enumerate(train_loader):

            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, targets)

            loss.backward()

            final_scores.update(targets, outputs.detach())

            summary_loss.update(loss.detach().item(), images.shape[0])

            optimizer.step()



            print("Train Step {:d}/{:d} \t summary_loss: {:.5f} \t final_score: {:.5f}".format(

                iter + 1, len(train_loader), summary_loss.avg, final_scores.avg

            ))



        print("[RESULT]: Train. Epoch: {:d} \t summary_loss: {:5f} \t final_score: {:.5f}".format(

            epoch + 1, summary_loss.avg, final_scores.avg

        ))

        scheduler.step(metrics=summary_loss.avg)

        val_loss = LossMeter()

        val_score = AccMeter()



        for step, (images, targets) in enumerate(valid_loader):

            with torch.no_grad():

                images, targets = images.to(device), targets.to(device)

                outputs = model(images)

                loss = criterion(outputs, targets)

                val_score.update(targets, outputs)

                val_loss.update(loss.detach().item(), images.shape[0])



        print("[RESULT]: Vaild. Epoch: {:d} \t summary_loss: {:5f} \t final_score: {:.5f}".format(

            epoch + 1, val_loss.avg, val_score.avg

        ))



    torch.save(model.state_dict(), '/kaggle/working/last.pt')



    return model



def test(model, test_loader):

    model.eval()

    with torch.no_grad():

        result = []

        for step, images in enumerate(test_loader):

            y_pred = model(images.to(device)).detach().cpu().numpy().argmax(axis=1).astype(int)

            result.extend(y_pred)

        sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

        sub['Label'] = result

        sub.to_csv('/kaggle/working/submission.csv', index=False)





train_data = DigitDataset(X_train, y_train, transforms=get_train_transforms())

valid_data = DigitDataset(X_valid, y_valid, transforms=get_valid_transforms())

test_data = DigitTestDataset(test_data, transforms=get_valid_transforms())



train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)

valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False, num_workers=0)

test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)



model = EfficientNet.from_name('efficientnet-b2')

state_dict = torch.load('/kaggle/input/efficientnet-pth/efficientnet-b2-8bb594d6.pth')

state_dict.pop('_fc.weight')

state_dict.pop('_fc.bias')

model.load_state_dict(state_dict, strict=False)



model._fc = nn.Linear(in_features=1408, out_features=10, bias=True)

model.to(device)

model = train(model, train_loader, valid_loader)

test(model, test_loader)