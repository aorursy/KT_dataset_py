import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

from tqdm import tqdm



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%matplotlib inline
import time

import torchvision.models as models

from torchvision import transforms

import torch

import torch.nn as nn

import torch.nn.functional as F



from torch.utils.data import Dataset, DataLoader



import cv2
if torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")
BATCH_SIZE=32

IMG_SHAPE = (224,224)
# Dataset class

class DanceData(Dataset):

    def __init__(self, df, img_shape, DATA_PATH, num_classes, class2idx):

        self.images = df['Image'].values

        self.img_shape = img_shape

        self.DATA_PATH = os.path.join(DATA_PATH, 'train')

        self.categories = df['target'].values

        self.num_classes = num_classes

        self.class2idx = class2idx



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        image = cv2.imread(os.path.join(self.DATA_PATH, self.images[idx]))

        image = cv2.resize(image, self.img_shape, interpolation=cv2.INTER_LINEAR)

        catg = self.categories[idx]

        return (torch.tensor(image, dtype=torch.float).permute(2,0,1), torch.tensor(class2idx[catg], dtype=torch.long))
# resnet18
class ResNet(nn.Module):

    def __init__(self, num_class):

        super(ResNet, self).__init__()

        self.model = models.resnet18(pretrained=True)

        self.linear1 = nn.Linear(1000, 100)

        self.output = nn.Linear(100,num_class)

    def forward(self, x):

        x = self.model(x)

        x = F.relu(self.linear1(x))

        x = F.softmax(self.output(x))

        return x
def loss_fn(output, target):

    return nn.CrossEntropyLoss()(output, target)
def train_fn(model, data_loader, optimizer, scheduler=None):

    model.train()

    

    for i,(img, cls) in enumerate(tqdm(data_loader, total=len(data_loader))):

        img = img.to(device, dtype=torch.float)

        cls = cls.to(device, dtype=torch.long)

        

        optimizer.zero_grad()

        output = model(img)

        loss = loss_fn(output, cls)

        loss.backward()

        optimizer.step()
def eval_fn(model, data_loader):

    model.eval()

    losses = []

    total = 0

    correct = 0

    with torch.no_grad():

        for i,(img, cls) in enumerate(tqdm(data_loader, total=len(data_loader))):

            img = img.to(device, dtype=torch.float)

            cls = cls.to(device, dtype=torch.long)

            output = model(img)

            out_idx = torch.argmax(output, axis=-1).cpu().numpy()

#             print(out_idx)

            total+=len(out_idx)

#             print(cls.cpu().numpy().tolist())

            correct+=(cls.cpu().numpy()==out_idx).sum()

            loss = loss_fn(output, cls)

            losses.append(loss.item())

        return np.mean(losses), correct/total
DATA_PATH = '/kaggle/input/identify-dance-forms/dataset'

train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

print(train.shape, test.shape)



classes = train.target.unique()

idx2cls = {i:cls for i, cls in enumerate(classes)}

class2idx = {cls:i for i, cls in idx2cls.items()}



train_df, valid_df = train_test_split(train, test_size=0.2, stratify=train.target)

print(train_df.shape, valid_df.shape)



train_dataset = DanceData(train_df, IMG_SHAPE, DATA_PATH, len(classes), class2idx)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)





valid_dataset = DanceData(train_df, IMG_SHAPE, DATA_PATH, len(classes), class2idx)

valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = ResNet(len(classes))

model.to(device)
EPOCHS = 8

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
val_losses = []

for epoch in range(EPOCHS):

    train_fn(model, train_dataloader, optimizer)

    val_loss, val_acc = eval_fn(model, valid_dataloader)

    print("Epoch : {}, Val loss : {}, Val acc : {}".format(epoch, val_loss, val_acc))

    val_losses.append(val_loss)
def preprocess_prediction_image(img_name):

    image = cv2.imread(os.path.join(DATA_PATH, 'test', img_name))

    image = cv2.resize(image, IMG_SHAPE, interpolation=cv2.INTER_LINEAR)

    image = torch.tensor(image, dtype=torch.float).permute(2,0,1)

    return image.unsqueeze(0).to(device, dtype=torch.float)
test.head(2)
pred = [idx2cls[torch.argmax(model(preprocess_prediction_image(i))).item()] for i in test.Image.values]
test['target'] = pred
test.to_csv('/kaggle/working/submission.csv', index=False)