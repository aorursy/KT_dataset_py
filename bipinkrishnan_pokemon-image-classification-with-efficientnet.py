!pip install efficientnet_pytorch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import models
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from efficientnet_pytorch import EfficientNet

from pathlib import Path
from tqdm.notebook import tqdm
train_path = '../input/pokemonimagedataset/dataset/train/'
test_path = '../input/pokemonimagedataset/dataset/test/'
def get_img_path(path):
    img_path = []
    for p in Path(train_path).glob('*/*'):
        img_path.append(p)
        
    return img_path
def encode_target(path):
    target = []
    for p in Path(train_path).glob('*'):
        if p.stem == 'NidoranF':
          target.append(p.stem[:-1])
        else:
          target.append(p.stem)
    
    return target
class LoadData(Dataset):
    def __init__(self, img_path, target, transform=None):
        self.img_path = img_path
        self.target = target
        self.transform = transform
        
    def __len__(self): return len(img_path)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx]).convert('RGB')
        target = self.target.index(Path(self.img_path[idx]).stem.split('.')[0])
        
        if self.transform:
            img = self.transform(img)
            
            return img, target
        else:
            return img, target
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5))
])

img_path = get_img_path(train_path)
target = encode_target(train_path)

val_img_path = get_img_path(test_path)
val_target = encode_target(test_path)
ds = LoadData(img_path, target, transform=transform)
ds_val = LoadData(val_img_path, val_target, transform=transform)
for i, (x, y) in enumerate(ds):
    print(x.shape, y, '------ training set')
    if i ==6:
      break
    
for j, (q, w) in enumerate(ds_val):
    print(q.shape, w, '------ valid set')
    if j ==6:
      break
bs = 16

trainloader = DataLoader(ds, batch_size=bs, shuffle=True)
testloader = DataLoader(ds_val, batch_size=bs, shuffle=True)
for i, (x, y) in enumerate(trainloader):
    print(x.shape, y.shape, '------ training set')
    if i ==8:
      break
    
for j, (q, w) in enumerate(testloader):
    print(q.shape, w.shape, '------ valid set')
    if j ==8:
      break
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = EfficientNet.from_pretrained("efficientnet-b2")
for param in model.parameters():
    param.requires_grad = False
    
model._fc = nn.Linear(1408, len(target))

for param in model.parameters():
    if param.requires_grad == True:
        print(param.shape)
model = model.to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CyclicLR(opt, base_lr=1e-3, max_lr=0.01, cycle_momentum=False)
def validate(dataloader):
  model.eval()
  total, correct = 0, 0
  for data in tqdm(dataloader, total=len(dataloader), leave=False):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, pred = torch.max(outputs, 1)

    total += labels.size(0)
    correct += (pred == labels).sum().item()

  return criterion(outputs, labels), (correct/total * 100)
epochs = 3

for epoch in range(epochs):
    model.train()
    for data, label in tqdm(trainloader, total=len(trainloader), leave=False):      
        opt.zero_grad()
        
        out = model(data.to(device))
        loss = criterion(out, label.to(device))
        loss.backward()
        
        opt.step()
        scheduler.step()
    
    validation = validate(testloader)
    
    print(f"Epoch: {epoch+1}/{epochs}\ttrain_loss: {loss.item()}\tval_loss: {validation[0].item()}\tval_acc: {validation[1]}")
for params in model.parameters():
    params.requires_grad = True
opt1 = optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
acc = 0.0

for epoch in range(epochs):
    model.train()
    for data, label in tqdm(trainloader, total=len(trainloader), leave=False):      
        opt1.zero_grad()
        
        out = model(data.to(device))
        loss = criterion(out, label.to(device))
        loss.backward()
        
        opt1.step()
    
    validation = validate(testloader)
    
    if validation[1]>acc:
        acc = validation[1]
        torch.save(model.state_dict(), f'./model-{round(validation[1], 4)}.pt')
    
    print(f"Epoch: {epoch+1}/{epochs}\ttrain_loss: {loss.item()}\tval_loss: {validation[0].item()}\tval_acc: {validation[1]}")