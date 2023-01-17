import gc

import os

import sys

import time

import random

import logging

import datetime as dt

import cv2



import numpy as np

import pandas as pd



import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils.data as data

import torch.nn.functional as F

import torchvision as vision

from torch.utils.data import Dataset, DataLoader



from torch.optim.lr_scheduler import CosineAnnealingLR



from pathlib import Path

from PIL import Image

from contextlib import contextmanager



from joblib import Parallel, delayed

from tqdm import tqdm

from fastprogress import master_bar, progress_bar



from sklearn.model_selection import KFold

from sklearn.metrics import fbeta_score
labels = pd.read_csv("../input/imet-2019-fgvc6/labels.csv")

train = pd.read_csv("../input/imet-2019-fgvc6/train.csv")  #109237

test = pd.read_csv("../input/imet-2019-fgvc6/sample_submission.csv")



#parameters

num_epochs = 200

batch_size = 200

learning_rate = 0.0001



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_transforms = {'train':vision.transforms.Compose([

        vision.transforms.RandomResizedCrop(332),

        vision.transforms.RandomHorizontalFlip(),

        vision.transforms.ToTensor(),

        vision.transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ]),

    'val': vision.transforms.Compose([

        vision.transforms.Resize(500),

        vision.transforms.RandomResizedCrop(332),

        vision.transforms.ToTensor(),

        vision.transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ])

}

data_transforms["test"] = data_transforms["val"]
class IMetDataset(Dataset):  

    def __init__(self, df, images_dir, transforms = None):

        self.df = df

        self.images_dir = images_dir

        self.transforms = transforms

    

    def __len__(self):

        return self.df.shape[0]

    

    def __getitem__(self, idx):

        cur_idx_row = self.df.iloc[idx]

        img_id = cur_idx_row['id']

        img_name = img_id + ".png"

        img_path = os.path.join(self.images_dir, img_name)

        

        img = cv2.imread(img_path)

        img = Image.fromarray(img)

        

        if self.transforms is not None:

            img = self.transforms(img)

        

        labels_list = cur_idx_row['attribute_ids'].split(' ')



        FILLVAL = 1.0

        label = torch.zeros((1103,), dtype=torch.float32)



        for i in labels_list:



            label[int(i)] = FILLVAL



        return img, label

    

class IMetLoadData(Dataset):

    def __init__(self, train, labels):

        self.train = train

        self.labels = labels

    

    def __len__(self):

        return self.train.shape[0]

    

    def __getitem__(self, idx):

        return self.train[idx], self.labels[idx]
class Classifier(nn.Module):

    def __init__(self):

        super(Classifier, self).__init__()

        

    def forward(self, x):

        return x



class Densenet201(nn.Module):

    def __init__(self, pretrained: Path):

        super(Densenet201, self).__init__()

        self.densenet201 = vision.models.densenet201()

        self.densenet201.load_state_dict(torch.load(pretrained))

        self.densenet201.classifier = Classifier()

        

        dense = nn.Sequential(*list(self.densenet201.children())[:-1])

        for param in dense.parameters():

            param.requires_grad = False

        

    def forward(self, x):

        return self.densenet201(x)

    

class MultiLayer(nn.Module):

    def __init__(self):

        super(MultiLayer, self).__init__()

        self.linear1 = nn.Linear(1920, 1920)

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(1920, 1103)

        self.dropout = nn.Dropout(0.5)

        self.sigmoid = nn.Sigmoid()

        

    def forward(self, x):

        x = self.relu(self.linear1(x))

        x = self.dropout(x)

        return self.sigmoid(self.linear2(x))
train_dataset = IMetDataset(train, "../input/imet-2019-fgvc6/train", 

                            transforms = data_transforms['train'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 

                          num_workers=2, pin_memory=True)

test_dataset = IMetDataset(test, "../input/imet-2019-fgvc6/test", 

                            transforms = data_transforms['test'])

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 

                          num_workers=2, pin_memory=True)



def get_feature_vector(df, loader):

    matrix = torch.zeros((df.shape[0], 1920)).to(device)

    preds = torch.zeros((df.shape[0], 1103)).to(device)

    model = Densenet201("../input/densenet201/densenet201.pth")

    model.to(device)

    batch = loader.batch_size

    for i, (i_batch,labels) in tqdm(enumerate(loader)):

        i_batch = i_batch.to(device)

        labels = labels.to(device)

        pred = model(i_batch).detach()

        matrix[i * batch:(i + 1) * batch] = pred

        preds[i * batch:(i + 1) * batch] = labels

    return matrix, preds



train_tensor, train_labels = get_feature_vector(train, train_loader)

test_tensor, test_labels = get_feature_vector(test, test_loader)
train_preds = np.zeros((len(train_tensor), 1103))

fold = KFold(n_splits = 5, random_state = 10)

loss_fn = nn.BCELoss(reduction="mean").to(device)

model = MultiLayer()

model.to(device)

tag = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

path = Path(f"bin/{tag}")

path.mkdir(exist_ok=True, parents=True)



for fold_num, (trn_idx, val_idx) in enumerate(fold.split(train_tensor)):

    X_train, X_val = train_tensor[trn_idx, :], train_tensor[val_idx, :]

    Y_train, Y_val = train_labels[trn_idx, :], train_labels[val_idx, :]

    

    X_train.to(device)

    Y_train.to(device)

    train_dataset = IMetLoadData(X_train, Y_train)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = IMetLoadData(X_val, Y_val)

    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    

    

        

    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_score = np.inf

    

    for epoch in range(num_epochs):

        model.train()

        avg_loss = 0.0

        

        for i_batch, y_batch in train_loader:

            y_pred = model(i_batch)

            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

        

        model.eval()

        valid_preds = np.zeros((len(val_loader.dataset), 1103))

        avg_val_loss = 0.0

        for i, (i_batch, y_batch) in enumerate(val_loader):

            with torch.no_grad():

                y_pred = model(i_batch).detach()

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(val_loader)

                valid_preds[i * batch_size:(i + 1) * batch_size] = y_pred.cpu().numpy()

                

        scheduler.step()

        

        print("=========================================")

        print(f"Epoch {epoch + 1} / {num_epochs}  Fold {fold_num + 1} / 5")

        print("=========================================")

        print(f"avg_loss: {avg_loss:.8f}")

        print(f"avg_val_loss: {avg_val_loss:.8f}")

            

        if best_score > avg_val_loss:

            torch.save(model.state_dict(), path / f"best{fold_num}.pth")

            best_score = avg_val_loss

        

    model.load_state_dict(torch.load(path / f"best{fold_num}.pth"))

    model.eval()

    valid_preds = np.zeros((len(val_loader.dataset), 1103))

    avg_val_loss = 0.0

    for i, (i_batch, y_batch) in enumerate(val_loader):

        with torch.no_grad():

            y_pred = model(i_batch).detach()

            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(val_loader)

            valid_preds[i * batch_size:(i + 1) * batch_size] = y_pred.cpu().numpy()

    print(f"Best Validation Loss: {avg_val_loss:.8f}")

    

    train_preds[val_idx] = valid_preds
def threshold_search(y_pred, y_true):

    score = []

    candidates = np.arange(0, 1.0, 0.01)

    for th in progress_bar(candidates):

        yp = (y_pred > th).astype(int)

        score.append(fbeta_score(y_pred=yp, y_true=y_true, beta=2, average="samples"))

    score = np.array(score)

    pm = score.argmax()

    best_th, best_score = candidates[pm], score[pm]

    return best_th, best_score



best_threshold, best_score = threshold_search(train_preds, train_labels.cpu().numpy())

print(f"best_threshold = {best_threshold}")

print(f"best_score = {best_score}")
test_dataset = IMetLoadData(test_tensor, test_labels)

test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

preds = np.zeros((test_tensor.size(0), 1103))



for pth in path.iterdir():

    model.load_state_dict(torch.load(pth))

    model.to(device)

    model.eval()

    temp = np.zeros_like(preds)

    for i, (i_batch, labels) in enumerate(test_loader):

        with torch.no_grad():

            y_pred = model(i_batch).detach()

            temp[i * batch_size:(i + 1) * batch_size] = y_pred.cpu().numpy()

    preds += temp / 5



preds = (preds > best_threshold).astype(int)

prediction = []

for i in range(preds.shape[0]):

    pred1 = np.argwhere(preds[i] == 1.0).reshape(-1).tolist()

    pred_str = " ".join(list(map(str, pred1)))

    prediction.append(pred_str)



test.attribute_ids = prediction

test.to_csv("submission.csv", index=False)

test.head()