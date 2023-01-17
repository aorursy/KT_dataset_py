import os
import zipfile
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
PATH = '../input/dogs-vs-cats-redux-kernels-edition/'
TRAIN_PATH = os.path.join(PATH, 'train.zip')
TEST_PATH = os.path.join(PATH, 'test.zip')

with zipfile.ZipFile(TRAIN_PATH, 'r') as z:
    z.extractall('.')
    
with zipfile.ZipFile(TEST_PATH, 'r') as z:
    z.extractall('.')
extract_label = lambda img_name: img_name.split('.')[0]
class CatsDogsDataset(Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform
        
    def __getitem__(self, index):
        img_name = self.img_list[index]
        
        image = Image.open('train/' + img_name)
        if self.transform:
            image = self.transform(image)
        
        label_img = extract_label(img_name)
        label = 1 if label_img == 'dog' else 0
        
        return image, label
    
    def __len__(self):
        return len(self.img_list)
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img_list = os.listdir('train/')
dataset = CatsDogsDataset(img_list=img_list, transform=data_transform)

train_size = 20000
val_size = 5000

train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)
for imgs, _ in train_loader:
    print('Image shape: ', imgs.shape)
    plt.figure(figsize=(16, 10))
    plt.axis('off')
    plt.imshow(make_grid(imgs, nrow=16, normalize=True).permute(1, 2, 0))
    break
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet50(pretrained=True)
model
for param in model.parameters():
    param.requires_grad = False

fc_in_features = model.fc.in_features
model.fc = nn.Linear(fc_in_features, 2)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

dataloaders = {'train': train_loader,
                'val': val_loader}

dataset_sizes = {'train': len(train_data),
                 'val': len(val_data)}

def train_model(model, criterion, optimizer, dataloaders, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

trained_model = train_model(model, criterion, optimizer, dataloaders, scheduler=lr_scheduler, num_epochs=5)
test_imgs = os.listdir('test/')

labels = []

with torch.no_grad():
    for test_img in test_imgs:
        img = Image.open('test/'+test_img)
        img = data_transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        
        trained_model.eval()
        output = trained_model(img)
        pred = F.softmax(output, dim=1)[:, 1].tolist()
        
        labels.append(pred[0])
df = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
df['label'] = labels
df.head()
df.to_csv('subm_2.csv', index=False)
