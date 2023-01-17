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
# ======================

# Params

BATCH_SIZE = 32

N_WORKERS = 4

N_EPOCHS = 4



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

class BengaliModel(torch.nn.Module):



    def __init__(self):

        super().__init__()

        self.backbone = torchvision.models.resnet18(pretrained=False)

        self.backbone.load_state_dict(torch.load("/kaggle/input/resnet18/resnet18.pth"))



        in_features = self.backbone.fc.in_features



        self.fc_graph = torch.nn.Linear(in_features, 168)

        self.fc_vowel = torch.nn.Linear(in_features, 11)

        self.fc_conso = torch.nn.Linear(in_features, 7)



    def forward(self, x):



        x = self.backbone.conv1(x)

        x = self.backbone.bn1(x)

        x = self.backbone.relu(x)

        x = self.backbone.maxpool(x)



        x = self.backbone.layer1(x)

        x = self.backbone.layer2(x)

        x = self.backbone.layer3(x)

        x = self.backbone.layer4(x)



        x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)



        fc_graph = self.fc_graph(x)

        fc_vowel = self.fc_vowel(x)

        fc_conso = self.fc_conso(x)



        return fc_graph, fc_vowel, fc_conso

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



device = torch.device("cuda:0")

model = BengaliModel()

model = model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()

plist = [{'params': model.parameters(), 'lr': 2e-5}]

optimizer = optim.Adam(plist, lr=2e-5)
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



torch.save(model.state_dict(), './baseline_weights.pth')