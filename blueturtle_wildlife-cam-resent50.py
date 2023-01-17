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
#Load data paths
train_image_path = "/kaggle/input/iwildcam-2020-fgvc7/train/"
test_image_path = "/kaggle/input/iwildcam-2020-fgvc7/test/"
train_annotations_path = "/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json"
#Read annotations JSON

import json

with open(train_annotations_path) as json_file:
    train_annotations = json.load(json_file)
#print(train_annotations.keys())
#print(train_annotations["annotations"])
categories = pd.DataFrame.from_dict(train_annotations["categories"])
categories
train_annotations["images"][1]
data = pd.DataFrame.from_dict(train_annotations["images"])
category_annotations = pd.DataFrame.from_dict(train_annotations["annotations"])
data = data.rename(columns = {"id": "image_id"})
data = data.merge(category_annotations, on = ("image_id"))
data = data.drop(columns = ["id", "count", "frame_num", "seq_num_frames", "seq_id"], axis = 1)
data
import matplotlib.pyplot as plt

img_path = (train_image_path + data["file_name"][1])
img = plt.imread((train_image_path + data["file_name"][1]))
plt.imshow(img)
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision
from torchvision import transforms
import tensorflow as tf


def transformer():
    return transforms.Compose([
        transforms.ToTensor()
    ])

class DataCreation(Dataset):
    def __init__(self, data, transforms = None):
        super().__init__()

        self.transform = transformer
        self.image_id = data["image_id"]

    def __len__(self):
        return len(self.image_id)
    
    def __getitem__(self,idx : int):
        image_id = self.image_id[idx]
        image = cv2.imread(train_image_path + image_id + ".jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
            
        return image_id, image
    

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = DataCreation(data, transformer())
train_loader = DataLoader(train_dataset, batch_size = 4, collate_fn = collate_fn)
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2024, 512),
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
    for image, image_id in train_loader:
        steps += 1
        image_id = (image_id + ".jpg")
        image_id = image_id.to(device)
        image = image.to(device)
    
        optimizer.zero_grad()
        logps = model.forward(image)
        loss = criterion(logps, image_id)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'aerialmodel.pth')