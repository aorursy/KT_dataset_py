import pandas as pd

import random

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

%matplotlib inline



from sklearn.model_selection import train_test_split



import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



import os

print(os.listdir("../input"))
# load data

warnings.filterwarnings("ignore")

df = pd.read_csv('../input/train.csv')

df.head()
len(df)
ix = random.randint(0, len(df)-1)

label, pixels = df.loc[ix][0], df.loc[ix][1:]

img = np.array(pixels).reshape((28,28))

print('label: ' + str(label))

plt.imshow(img)
# transforming df for easier manipulation

def transform_df(df):

    labels, imgs = [], []

    for index, row in df.iterrows():

        label, pixels = row[0], row[1:]

        img = np.array(pixels)

        labels.append(label)

        imgs.append(img)



    df_img = pd.DataFrame({'label': labels, 'img': imgs})

    # to speed up the process we can use for example only 1000 samples

    # df_img = df_img[:1000]

    return df_img



df_img = transform_df(df)

df_img.head()
# checking images using new df structure

ix = random.randint(0, len(df_img)-1)

img = df_img.loc[ix].img.reshape((28,28))

label = df_img.loc[ix].label

print('label: ' + str(label))

plt.imshow(img)
train_df, test_df = train_test_split(df_img, test_size=0.2, shuffle=True)

print(len(train_df), len(test_df))
train_df.head()
# create torch dataset

from torch.utils.data import Dataset

class MNISTDataset(Dataset):

  def __init__(self, imgs, labels):    

    super(MNISTDataset, self).__init__()

    self.imgs = imgs

    self.labels = labels

  def __len__(self):

    return len(self.imgs)

  def __getitem__(self, ix):

    img = self.imgs[ix]

    label = self.labels[ix]

    return torch.from_numpy(img).float(), label



dataset = {

    'train': MNISTDataset(train_df.img.values, train_df.label.values),

    'test': MNISTDataset(test_df.img.values, test_df.label.values)

} 



len(dataset['train'])
# again checking image, now based on torch dataset

ix = random.randint(0, len(dataset['train'])-1)

img, label = dataset['train'][ix]

print(img.shape, img.dtype)

print(label)

plt.imshow(img.reshape((28,28)))
# create model

import torch.nn as nn

def block(in_f, out_f):

  return nn.Sequential(

      nn.Linear(in_f, out_f),

      nn.BatchNorm1d(out_f),

      nn.ReLU(inplace=True),

      #nn.Dropout(),

  )

model = nn.Sequential(

  block(784,512),

  block(512,256),

  block(256,128),

  nn.Linear(128, 10)

)

model.to(device)
from torch.utils.data import DataLoader

import torch.optim as optim

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.1)

scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=3, min_lr=0.0001, verbose=True)



dataloader = {

    'train': DataLoader(dataset['train'], batch_size=32, shuffle=True, num_workers=4),

    'test': DataLoader(dataset['test'], batch_size=32, shuffle=False, num_workers=4),

}
# train

best_acc, stop, early_stop = 0, 0, 10

for e in range(100):



    model.train()

    total_loss = []

    for imgs, labels in tqdm(dataloader['train']):

        imgs, labels = imgs.to(device), labels.to(device)

        preds = model(imgs)

        optimizer.zero_grad()

        loss = criterion(preds, labels)

        loss.backward()

        optimizer.step()

        total_loss.append(loss.data)



    model.eval()

    val_loss, acc = [], 0.

    with torch.no_grad():

        for imgs, labels in tqdm(dataloader['test']):

            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)

            loss = criterion(preds, labels)

            val_loss.append(loss.data)

            _, preds = torch.max(preds, 1)

            acc += (preds == labels).sum().item()



    acc /= len(dataset['test'])

    if acc > best_acc:

        print('\n Best model ! saved.')

        torch.save(model.state_dict(), 'best_model.pt')

        best_acc = acc

        stop = -1



    stop += 1

    if stop >= early_stop:

        break



    scheduler.step(acc)



    print('\n Epoch {}, Training loss: {:4f}, Val loss: {:4f}, Val acc: {:4f}'.format(

        e + 1, torch.mean(torch.stack(total_loss)), torch.mean(torch.stack(val_loss)), acc))



print('\n Best model with acc: {}'.format(best_acc))
# test

model.load_state_dict(torch.load('best_model.pt'))

model.to(device)

model.eval()



ix = random.randint(0, len(dataset['test'])-1)

img, label = dataset['test'][ix]

pred = model(img.unsqueeze(0).to(device)).cpu()

pred_label = torch.argmax(pred)

print('Ground Truth: {}, Prediction: {}'.format(label, pred_label))

plt.imshow(img.reshape((28,28)))
submission = pd.read_csv('../input/test.csv')

submission.head()
imgs = []

for index, row in submission.iterrows():

    pixels = row[0:]

    img = np.array(pixels)

    imgs.append(img)



submission_transf = pd.DataFrame({'img': imgs})

submission_transf.head()
# converting into pytorch dataset

# inserting index values as labels

submission_pt = {

    'test': MNISTDataset(submission_transf.img.values, submission_transf.index.values)

} 
# test individual samples from dropout dataset

model.load_state_dict(torch.load('best_model.pt'))

model.to(device)

model.eval()



ix = random.randint(0, len(dataset['test'])-1)

img, idx = submission_pt['test'][ix]

pred = model(img.unsqueeze(0).to(device)).cpu()

pred_label = torch.argmax(pred)

print(type(idx))

print('Prediction: {}'.format(pred_label))

plt.imshow(img.reshape((28,28)))
# make predictions on every image

subm_dict = dict()



for ix in range(0,len(submission_pt['test'])):

    img, idx = submission_pt['test'][ix]

    pred = model(img.unsqueeze(0).to(device)).cpu()

    pred_label = torch.argmax(pred)

    subm_dict[idx+1] = pred_label.item()
# create submission file

final_df = pd.DataFrame.from_dict(subm_dict, orient='index')

final_df.index.name = 'ImageId'

final_df.columns = ['Label']

final_df.to_csv('submission.csv')