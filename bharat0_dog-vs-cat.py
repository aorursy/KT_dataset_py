import numpy as np
import pandas as pd
import os
import glob

from PIL import Image

import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import matplotlib.pyplot as plt
torch.manual_seed(1)
train_images = list(glob.iglob('../input/train/*.jpg'))[:300]
img_path = np.random.choice(train_images)
img = Image.open(img_path)
print(img.size)
img
transforms = trans.Compose([
    trans.Resize((64, 64)),
    trans.ToTensor()
])
X = transforms(img)
X = X.view(1, 3, 64, 64)
Y = torch.FloatTensor([ 1 if 'dog' in img_path else 0 ])
Y
plt.imshow(X[0].numpy().transpose(1,2,0))
class DogCatDataset:
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        img_path = self.images[i]
        img = Image.open(img_path)
        
        y = 1 if 'dog' in img_path else 0
        
        if self.transform:
            img = self.transform(img)
        return img, y
train_dataset = DogCatDataset(train_images, transform=transforms)
train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
class DCClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.p1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.p2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.p3 = nn.MaxPool2d(2)
        
        self.l1 = nn.Linear(64 * 8 * 8, 512)
        self.l2 = nn.Linear(512, 1)
    
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.p1(self.conv1(x)))
        x = self.relu(self.p2(self.conv2(x)))
        x = self.relu(self.p3(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.l1(x))
        x = self.sig(self.l2(x))
        return x
model = DCClassifier()
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()
def train(e):
    model.train()
    l = a = n = 0
    for i, (X, Y) in enumerate(train_dl):
        Y = Y.float()
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, Y.view_as(preds))
        loss.backward()
        optimizer.step()
        
        l += loss.item()
        a += ((preds.squeeze() > 0.5) == Y.byte()).sum().item() / Y.shape[0]
        n += 1
#         if (i+1) % 100 == 0:
    print(f"Epoch {e} Iter {i+1} Loss {l/n} Acc {a/n}")
    l = a = n = 0
for i in range(10):
    train(i)
model = DCClassifier()
optimizer = optim.RMSprop(model.parameters())
for i in range(10):
    train(i)
model = DCClassifier()
optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
for i in range(10):
    train(i)
class DCClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.p1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.p2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.p3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.p4 = nn.MaxPool2d(2)
        
        self.l1 = nn.Linear(128 * 4 * 4, 2048)
        self.l2 = nn.Linear(2048, 512)
        self.l3 = nn.Linear(512, 64)
        self.l4 = nn.Linear(64, 1)
    
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.p1(self.conv1(x)))
        x = self.relu(self.p2(self.conv2(x)))
        x = self.relu(self.p3(self.conv3(x)))
        x = self.relu(self.p4(self.conv4(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.sig(self.l4(x))
        return x
model = DCClassifier()
optimizer = optim.Adam(model.parameters())
for i in range(10):
    train(i)
model = DCClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
for i in range(10):
    train(i)
model = DCClassifier()
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
for i in range(10):
    train(i)
# def get_stats(m):
#     return [ m.mean().item(), m.median().item(), m.std().item(), m.min().item(), m.max().item() ]

# def print_model(model):
#     index = []
#     columns = ["Mean", "Median", "Std", "Min", "Max" ]
#     data = []
#     for n, m in model.named_parameters():
#         index.append(n)
#         index.append(f'{n}.grad')
#         data.append(get_stats(m))
#         if m.grad is not None:
#             data.append(get_stats(m.grad))
#         else:
#             data.append([ None ] * len(columns))
#     return pd.DataFrame(data, index=index, columns=columns)   
# print_model(model)
# plt.imshow(X[0].numpy().squeeze().transpose(1, 2, 0))

# optimizer.zero_grad()
# o = model(X)
# # print(o, Y)
# loss = criterion(o, Y.view_as(o))
# loss.backward()
# # print(model.conv1.weight.grad, model.l4.weight.grad)
# optimizer.step()
# print(loss)

# o = model(X)
# loss =criterion(torch.sigmoid(o), Y.view_as(o))

# loss

# loss.backward()

# ag = loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0] #.variable is model.l3.bias

# ag.variable.grad

# ag.variable

# model.l3.bias.grad

# optimizer.step()

# optimizer.zero_grad()

# print_model(model)


     

# print_model(model)

