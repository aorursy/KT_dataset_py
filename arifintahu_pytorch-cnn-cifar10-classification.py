!pip install py7zr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from PIL import Image

from py7zr import unpack_7zarchive
import shutil
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
shutil.unpack_archive('/kaggle/input/cifar-10/train.7z', '/kaggle/working')
train_dir = os.listdir("/kaggle/working/train")
train_dir_len = len(train_dir)
print(".\\train:\t",train_dir_len)
print("files:\t\t",train_dir[:3])
import pandas as pd
train_labels = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv',dtype=str)
train_images = pd.DataFrame(columns = ['id','label','path'],dtype=str)
test_labels = pd.read_csv('/kaggle/input/cifar-10/sampleSubmission.csv')
train_labels.info()
path_base = '/kaggle/working/train/'

for index in range(0,train_dir_len):
    path = path_base + str(index+1)+'.png'
    if os.path.exists(path):
        train_images = train_images.append([{ 'id': str(train_labels['id'].iloc[index]),'path': path, 'label':train_labels['label'].iloc[index]}])
        
train_images.head(2)
display_groupby = train_images.groupby(['label']).count()
display_groupby.head(10)
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
for name in  class_names:
    index = class_names.index(name)
    train_images.loc[train_images.label==name,'label'] = str(index)
train_images.head(2)
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            transforms.functional.to_tensor(Image.open(row["path"])),
            int(row["label"]),
        )

dataset = MyDataset(train_images)
BATCH_SIZE = 64
NUM_WORKERS = 0
VALIDATION_SIZE = 0.2
num = len(dataset)
split = round(num*VALIDATION_SIZE)

train_dataset, val_dataset = random_split(dataset, [num-split, split])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
def imshow(img):
    # convert from Tensor image
    plt.imshow(np.transpose(img, (1, 2, 0)))  
    
# Show Image Dataloader
train_dataiter = iter(train_loader)
images, labels = train_dataiter.next()
images = images.numpy()
fig = plt.figure(figsize=(25, 4))

# Display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title([labels[idx]])
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, input):
        x = self.pool(F.relu(self.conv1(input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Net()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 100
time0 = time()

train_loss_list = []
val_loss_list = []


if torch.cuda.is_available():
    for epoch in range(n_epochs):
        train_loss = 0.0
        val_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            
            output = model(images.to(device))
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        for images, labels in val_loader:
            output = model(images.to(device))
            loss = criterion(output, labels.to(device))
            
            val_loss += loss.item()

        train_loss = train_loss
        val_loss = val_loss
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        print('Epoch: {} \tTraining loss: {:.6f} \tValidation loss: {:.6f}'.format(
            epoch+1,
            train_loss,
            val_loss
        ))
    
print("\nTraining Time (in minutes) =",(time()-time0)/60)
plt.plot(range(100),train_loss_list)
plt.plot(range(100),val_loss_list)
to_pil = torchvision.transforms.ToPILImage()

images, labels = next(iter(val_loader))

img = images[1].view(1, 3, 32, 32).to(device)
with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
probab = list(ps.cpu().numpy()[0])

print(probab)
index_predict = probab.index(max(probab))
print(index_predict, class_names[index_predict])
print(labels[1])

img = to_pil(images[1])
plt.imshow(img)
correct_count, all_count = 0, 0
for images,labels in val_loader:
    for i in range(len(labels)):
        img = images[i].view(1, 3, 32, 32).to(device)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
