# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
!pip install jcopdl
import torch
from torch import nn, optim
from jcopdl.callback import Callback, set_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
!pip install gdown
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
bs = 128
crop_size = 224

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Grayscale(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(230),
    transforms.CenterCrop(crop_size),
    transforms.Grayscale(),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder("../input/chest-xray-pneumonia/chest_xray/train/", transform=train_transform)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)

test_set = datasets.ImageFolder("../input/chest-xray-pneumonia/chest_xray/test/", transform=test_transform)
testloader = DataLoader(test_set, batch_size=bs, shuffle=True)
feature, target = (next(iter(trainloader)))
feature.shape
label2cat = train_set.classes
label2cat
from jcopdl.layers import linear_block, conv_block
class CNN(nn.Module):
    """
    Input: (N, C, H, W)
    Output: (N, C)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(1, 7, batch_norm=True), # 112x112
            conv_block(7, 14, batch_norm=True), # 56x56
            conv_block(14, 28, batch_norm=True), # 28x28
            conv_block(28, 56, batch_norm=True), # 14x14
            conv_block(56, 112, batch_norm=True), # 7x7
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            linear_block(112*7*7, 784, batch_norm=True, dropout=0.2),
            linear_block(784, 2, activation="lsoftmax")
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
linear_block
config = set_config({
    "batch_size": bs,
    "crop_size": crop_size
})
model = CNN().to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
callback = Callback(model, config, outdir="model")
from tqdm.auto import tqdm

def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, device):
    if mode == "train":
        model.train()
    elif mode == "test":
        model.eval()
    cost = correct = 0
    for feature, target in tqdm(dataloader, desc=mode.title()):
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)
        
        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        cost += loss.item() * feature.shape[0]
        correct += (output.argmax(1) == target).sum().item()
    cost = cost / len(dataset)
    acc = correct / len(dataset)
    return cost, acc
while True:
    train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
    with torch.no_grad():
        test_cost, test_score = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)
    
    # Logging
    callback.log(train_cost, test_cost, train_score, test_score)

    # Checkpoint
    callback.save_checkpoint()
        
    # Runtime Plotting
    callback.cost_runtime_plotting()
    callback.score_runtime_plotting()
    
    # Early Stopping
    if callback.early_stopping(model, monitor="test_score"):
        callback.plot_cost()
        callback.plot_score()
        break
feature, target = next(iter(testloader))
feature, target = feature.to(device), target.to(device)
with torch.no_grad():
    model.eval()
    output = model(feature)
    preds = output.argmax(1)
preds
fig, axes = plt.subplots(6, 6, figsize=(24, 24))
for image, label, pred, ax in zip(feature, target, preds, axes.flatten()):
    ax.imshow(image.permute(1, 2, 0).cpu())
    font = {"color": 'r'} if label != pred else {"color": 'g'}        
    label, pred = label2cat[label.item()], label2cat[pred.item()]
    ax.set_title(f"L: {label} | P: {pred}", fontdict=font);
    ax.axis('off');
