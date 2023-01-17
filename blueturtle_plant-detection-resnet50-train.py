# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
"""for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))"""

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test_image_path = "/kaggle/input/herbarium-2020-fgvc7/nybg2020/test/"
import json                                           # Loading Metadata

train_path = "/kaggle/input/herbarium-2020-fgvc7/nybg2020/train/"

with open((train_path + "metadata.json"),"r", encoding = "ISO-8859-1") as file:
    #dict_keys(['annotations', 'categories', 'images', 'info', 'licenses', 'regions'])
    metadata = json.load(file)

categories = pd.DataFrame.from_dict(metadata["annotations"])
metadata = pd.DataFrame.from_dict(metadata["images"])

data = pd.merge(metadata, categories, on = ("id"))
data = data.drop(columns = "id")
data

#data[data["category_id"] == 23079]
import matplotlib.pyplot as plt

#img = plt.imread((train_path + "images/156/72/354106.jpg"))
#plt.imshow(img)
import torchvision
from torchvision import transforms
import albumentations as A #Package of transformations
from albumentations.pytorch.transforms import ToTensorV2

def train_transform():
    return A.Compose([
        ToTensorV2(p = 1),
    ])

def test_transform():
    return A.Compose([
        ToTensorV2(p = 1),
    ])
from torch.utils.data import Dataset, DataLoader
import cv2

class CreateDataset(Dataset):
    def __init__(self, data, labels, transforms = train_transform):
        self.data = data
        self.transform = transforms
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        file_name = self.data["file_name"].values[index]
        image = cv2.imread(train_path + file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        label = self.labels.values[index]
        print("Label type before transform loop: " + str(type(label))) #Label is a numpy array of int type
        print("Image type before transforms loop: "+ str(type(image)))#image is a numpy array
        if self.transform:
            image = {"image" : image,}
            image = self.transform(**image) 
            print("Label type after transform loop: " + str(type(label))) 
            print("Image type after transform loop: " + str(type(image)))
                 
        return image, label
    
from sklearn.model_selection import train_test_split

def collate_fn(batch):
    return (zip(*batch))

train_data, test_data = train_test_split(data, train_size = 0.8)

train_dataset = CreateDataset(train_data, train_data["category_id"], train_transform())
test_dataset = CreateDataset(test_data, test_data["category_id"], test_transform())

train_loader = DataLoader(train_dataset, batch_size = 4, collate_fn = collate_fn)
test_loader = DataLoader(test_dataset, batch_size = 4, collate_fn = collate_fn)
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)
epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    for image, label in train_loader:
        steps += 1
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        logps = model.forward(image)
        loss = criterion(logps, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, category in test_loader:
                    images, category = images.to(device), category.to(device)
                    logps = model.forward(images)
                    batch_loss = criterion(logps, category)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == category.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(test_loader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'PlantDetectionResNet50.pth')