import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import re
import glob

import copy

np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### ★★★ 画像データのディレクトリ, 事前学習する/しない(True/False), とエポック数を入れてください.  ★★★
dirname = '../input/sozai6mini/sozai6/'
PRETRAINED = False
EPOCHS = 1

1+2
print(dirname)

path_images = os.path.join(dirname, '*/*.jpg')
path_images = glob.glob(path_images)
n_samples = 500
#path_images = list(np.random.choice(path_images, n_samples))
path_images = path_images[:n_samples]
path_images.sort()
print('Num. of files: {}'.format(len(path_images)))
path_images


def ds_split(ds, **kwrgs):
  import numpy as np
  import pandas as pd
  ds=pd.DataFrame(ds)
  N = len(ds)
  dataset = {}
  index = ds.index
  for key in kwrgs.keys():
    p = kwrgs[key]
    n = int(N * p)
    ids = list(np.random.choice(a=index, size=n, replace=False))
    dataset[key] = ds.loc[ids]
    index = list(set(index) - set(ids))
  _dataset = ds.loc[index]
  dataset[key] = pd.concat([dataset[key], _dataset])
  for key in dataset.keys():
    dataset[key] = list(dataset[key].sort_index().iloc[:, 0])
  return dataset

path_images_ = ds_split(path_images, train=0.8, val=0.2)

class ImageDataset(Dataset):
  def __init__(self, path_images, transform=None):
    self.path_images = path_images
    self.labels = list(map(lambda x:x.split('/')[-2], path_images))
    self.transform = transform
    label_set = list(set(self.labels))
    self.label_set = dict(list(zip(label_set, np.arange(len(label_set)))))
  
  def __len__(self):
    return len(self.path_images)

  def __getitem__(self, idx):
    path_image = path_images[idx]
    label = self.path_images[idx].split('/')[-2]
    onehot_labels = np.zeros([len(self.labels), len(self.label_set)])
    onehot_label  = onehot_labels[idx]
    onehot_label[self.label_set[label]] = 1.
    onehot_label = torch.tensor(onehot_label)
    image = Image.open(path_image)
    resize = np.array(image.size) * 0.47
    resize = tuple(resize.astype(int))
    image = image.resize(resize)

    if self.transform:
      image = self.transform(image)
      #image = torch.tensor
    
    return [onehot_label, image]

ds = ImageDataset(path_images)

sample_ids = np.random.choice(np.arange(ds.__len__()), 5)
for sample_id in sample_ids:
  label, image = ds[sample_id]
  print('Label: {}'.format(ds.labels[sample_id]))
  print('One-hot: {}'.format(label))
  print('Image: {}'.format(image.size))
  plt.imshow(image)
  plt.show()

data_transform = transforms.Compose(
  [
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]
)

ds = ImageDataset(path_images, data_transform)

for sample_id in sample_ids:
  label, image = ds[sample_id]
  print('Label: {}'.format(ds.labels[sample_id]))
  print('One-hot: {}'.format(label))
  pilTrans = transforms.ToPILImage()
  pilImg = pilTrans(image)
  print('Image: {}'.format(pilImg.size))
  plt.imshow(pilImg)
  plt.show()
ds_ = {}
dataset_sizes_ = {}
for phase in ['train', 'val']:
  ds_[phase] = ImageDataset(path_images_[phase], data_transform)
  dataset_sizes_[phase] = len(ds_[phase])

dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=10)
n_labels = len(dl.dataset.label_set)

dl_ = {}
for phase in ['train', 'val']:
  dl_[phase] = DataLoader(ds_[phase], batch_size=32, shuffle=True, num_workers=10)

class Net(nn.Module):
  def __init__(self, pretrained=False):
    super(Net, self).__init__()
    resnet = models.resnet18(pretrained=pretrained)
    self.resnet = nn.Sequential(*list(resnet.children())[:-1])
    self.image_fc = nn.Linear(512, 100)
    self.fc = nn.Linear(100, n_labels)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    
  def forward(self, image):
    x = self.resnet(image).view(-1, 512)
    x = self.relu(self.image_fc(x))
    ret = self.fc(x)
    ret = self.sigmoid(ret)
    return ret
  
model = Net(pretrained=PRETRAINED)
model = model.to(device)

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# criterion
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()

def train_model(dataset_sizes, model, criterion, optimizer, dataloaders, num_epochs=1):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  # best_acc = 0.0
  best_loss = 10000000

  train_loss = []
  val_loss = []
  train_acc = []
  val_acc = []
  
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    
    label_list = torch.tensor([]).to(device)
    output_list = torch.tensor([]).to(device)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        print(phase)
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0

        # Iterate over data.
        for onehot_labels, images in dataloaders[phase]:

          onehot_labels = onehot_labels.to(device)
          images = images.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward
          # track history if only in train
          with torch.set_grad_enabled(phase == 'train'):
            outputs = model(images)
            #loss = criterion(outputs, torch.tensor(onehot_labels.float(), device=device).view(-1, 1))
            loss = criterion(outputs, torch.tensor(onehot_labels.float(), device=device))

            # backward + optimize only if in training phase
            if phase == 'train':
              loss.backward()
              optimizer.step()

          #label_list = torch.cat((label_list, onehot_labels.float().view(-1, 1)))
          label_list = torch.cat((label_list, onehot_labels.float()))
          output_list = torch.cat((output_list, outputs))

          # statistics
          running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / dataset_sizes[phase]

        #epoch_acc = corref(label_list, output_list).item()
        if phase == 'train':
          train_loss.append(epoch_loss)
          #train_acc.append(epoch_acc)
        else:
          val_loss.append(epoch_loss)
          #val_acc.append(epoch_acc)

        print(
          '{} Loss: {:.4f} Corref: {:.4f}'.format(
            phase, 
            epoch_loss, 
            1
            #epoch_acc
          )
        )

        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
          best_loss = epoch_loss
          best_model_wts = copy.deepcopy(model.state_dict())
          
  return label_list, output_list
label_list, output_list = train_model(dataset_sizes_, model, criterion, optimizer, dl_, num_epochs=EPOCHS)
