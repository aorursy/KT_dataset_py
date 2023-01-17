!pip install pytorchcv
import os



import pandas as pd

import numpy as np



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import random_split, DataLoader, Dataset

import torchvision

from torchvision import transforms

from sklearn.model_selection import train_test_split

from pytorchcv.model_provider import get_model



import PIL

from tqdm import tqdm
mask_list = os.listdir('../input/face-mask-detection/dataset/with_mask')

without_mask_list = os.listdir('../input/face-mask-detection/dataset/without_mask')
total_list = [os.path.join('../input/face-mask-detection/dataset/with_mask', path) for path in os.listdir('../input/face-mask-detection/dataset/with_mask')] + [os.path.join('../input/face-mask-detection/dataset/without_mask', path) for path in os.listdir('../input/face-mask-detection/dataset/without_mask')]

label_list = [0 for i in range(len(mask_list))] + [1 for i in range(len(without_mask_list))]
df = pd.DataFrame({'Image': total_list})

df['Label'] = label_list

df
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)
class MaskDataset(Dataset):

    def __init__(self, df, transform = None):

        self.image = df['Image'].tolist()

        self.label = df['Label'].tolist()

        self.transform = transform

        

    def __getitem__(self, index):

        img_path = self.image[index]

        img = PIL.Image.open(img_path).convert('RGB')

        category = self.label[index]

        if self.transform:

            img = self.transform(img)

        

        return img, category

    

    def __len__(self):

        return len(self.label)
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



train_transform = transforms.Compose([

                    transforms.Resize((224, 224)),

                    transforms.RandomHorizontalFlip(),

                    transforms.ToTensor(),

                    transforms.Normalize(*imagenet_stats,inplace=True)

                  ])



valid_transform = transforms.Compose([

                    transforms.Resize((224, 224)),

                    transforms.ToTensor(),

                    transforms.Normalize(*imagenet_stats)

                  ])
class MaskModel(nn.Module):

    def __init__(self):

        super(MaskModel, self).__init__()

        # Load pretrained network as backbone

        pretrained = get_model('vgg16', pretrained=True)

        # remove last layer of fc

        self.backbone = pretrained.features

        self.fc3 = nn.Linear(25088, 2)

        

        nn.init.zeros_(self.fc3.bias)

        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.02)

        

        del pretrained



    def forward(self, x):

        x = self.backbone(x)

        x = x.reshape(x.size(0), -1)

        x = self.fc3(x)



        return x

    

    def freeze_backbone(self):

        """Freeze the backbone network weight"""

        for p in self.backbone.parameters():

            p.requires_grad = False

            

    def unfreeze_backbone(self):

        """Freeze the backbone network weight"""

        for p in self.backbone.parameters():

            p.requires_grad = True
train_ds = MaskDataset(train_df,  transform=train_transform)

valid_ds = MaskDataset(valid_df,  transform=valid_transform)
def accuracy(prediction, ground_truth):

    num_correct = (np.array(prediction) == np.array(ground_truth)).sum()

    return num_correct / len(prediction)
EPOCHS = 5 

BATCH_SIZE = 4

LR = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

print(device)

model = MaskModel().to(device)

model.freeze_backbone()

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

criterion = nn.CrossEntropyLoss()



train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

valid_dl = DataLoader(valid_ds, BATCH_SIZE, num_workers=4, pin_memory=True)
for epoch in range(EPOCHS):

    model.train()

    

    for img, label in tqdm(train_dl):

        img = img.to(device)

        label = label.to(device)

        optimizer.zero_grad()

        logits = model(img)

        loss = criterion(logits, label)

        loss.backward()

        optimizer.step()

        

    model.eval()

    

    predictions = []

    ground_truths = []

    for img, label in tqdm(valid_dl):

        img = img.to(device)

        with torch.no_grad():

            logits = model(img)

            prediction = torch.argmax(logits, dim=1)



            predictions.extend(prediction.tolist())

            ground_truths.extend(label.tolist())



    acc = accuracy(predictions, ground_truths)

    print(acc)