import torch
import os
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],
                                                                             std=[0.229,0.224,0.225])])

class MyDataset(Dataset):
  def __init__(self, image_dir, label, transforms=None):
    self.image_dir = image_dir
    self.label = label
    self.image_list = os.listdir(self.image_dir)
    self.transforms = transforms
  
  def __len__(self):
    return len(self.image_list)
  
  def __getitem__(self,idx):
    # if torch.is_tensor(idx):
    #   idx = idx.tolist()

    image_name = os.path.join(self.image_dir, self.image_list[idx])
    image = io.imread(image_name)

    ### transform
    image = transforms(image)

    return (image,self.label)

root = '/kaggle/input/'
inter = os.listdir(root)[0]
root = root + inter
#cheetah : 0 , jaguar : 1, tiger : 2, hyena : 3

cheetah_train = MyDataset(root+"/train/cheetah_train_resized",0,transforms)
jaguar_train = MyDataset(root+"/train/jaguar_train_resized",1,transforms)
tiger_train = MyDataset(root+"/train/tiger_train_resized",2,transforms)
hyena_train = MyDataset(root+"/train/hyena_train_resized",3,transforms)
train_set = ConcatDataset([cheetah_train, jaguar_train, tiger_train, hyena_train])
print("Number of Training set images : ", len(train_set))

cheetah_val = MyDataset(root+"/validation/cheetah_validation_resized",0, transforms)
jaguar_val = MyDataset(root+"/validation/jaguar_validation_resized",1, transforms)
tiger_val = MyDataset(root+"/validation/tiger_validation_resized",2, transforms)
hyena_val = MyDataset(root+"/validation/hyena_validation_resized",3,transforms)
val_set = ConcatDataset([cheetah_val, jaguar_val, tiger_val, hyena_val])
print("Numver of Validation set images : ", len(val_set))
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

### learning rate
LR = 0.001
EPOCH = 5
BATCH_SIZE = 32


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

model = models.resnet18(pretrained=False).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
def train(model, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (image, target) in enumerate(train_loader):
    data, target = image.to(DEVICE), target.to(DEVICE)
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0 :
      print('Train Epoch : {} [{}/{} ({:.0f})%]\tLoss: {:.6f}'
      .format(epoch, batch_idx*len(image),len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.item()))

def evaluate(model, test_loader):
  model.eval()
  test_loss =0
  correct =0
  with torch.no_grad():
    for (image, target) in test_loader:
      image, label = image.to(DEVICE), target.to(DEVICE)
      output = model(image)

      test_loss += F.cross_entropy(output, label, reduction='sum').item()
      pred = output.max(1, keepdim=True)[1]
      correct+= pred.eq(label.view_as(pred)).sum().item()
  
  test_loss /= len(test_loader.dataset)
  test_accuracy = 100. * correct / len(test_loader.dataset)
  return test_loss, test_accuracy
for epoch in range(1, EPOCH+1):
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, val_loader)
    print('[{}] Test Loss : {:.4f}, Accuracy : {:.4f}%'.format(epoch, test_loss, test_accuracy))
class TestDataset(Dataset):
  def __init__(self, image_dir, transforms=None):
    self.image_dir = image_dir
    self.image_list = os.listdir(self.image_dir)
    self.transforms = transforms
  
  def __len__(self):
    return len(self.image_list)
  
  def __getitem__(self,idx):
    # if torch.is_tensor(idx):
    #   idx = idx.tolist()

    image_name = os.path.join(self.image_dir, self.image_list[idx])
    image = io.imread(image_name)

    ### transform
    image = transforms(image)

    return (image,self.image_list[idx].split('.')[0])


root = '/kaggle/input/testoftest'

test_set = TestDataset(root, transforms)
test_loader = DataLoader(test_set)
import pandas as pd


#cheetah : 0 , jaguar : 1, tiger : 2, hyena : 3
map = ['cheetah','jaguar','tiger','hyena']

model.eval()
df = pd.DataFrame(columns=['id','category'])
with torch.no_grad():
    for (image, image_name) in test_loader:
        image = image.to(DEVICE)
        output = model(image)
        pred = output.max(1, keepdim=True)[1]
        df = df.append(pd.DataFrame([[image_name[0], map[pred.squeeze().tolist()]]], columns=['id','category']))

df

df.to_csv('/kaggle/working/'+'res.csv', index=False)