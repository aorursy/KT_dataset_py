import torch
import torch.nn as nn
import torch.functional as F
from torchvision import *
from torch.optim import *
from torch.utils.data import DataLoader

import pathlib

import time

import copy

import numpy as np

from PIL import Image
from PIL import ImageFile
import albumentations
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
device = get_default_device()
device
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder('../input/intel-image-classification/seg_train/seg_train', transform=train_transform)
test_dataset = datasets.ImageFolder('../input/intel-image-classification/seg_test/seg_test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
root = pathlib.Path('../input/intel-image-classification/seg_train/seg_train')
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        self.backbone =  models.vgg19(pretrained=True)
        num_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(num_features, len(classes))
        
    def forward(self,image):
        x = self.backbone(image)
        return x
model = Model()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.backbone.parameters(), lr=0.00001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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
def prediciton(net, data_loader):
    test_pred = torch.LongTensor()
    for i, data in enumerate(data_loader):
        if torch.cuda.is_available():
          pass

        output = net(data)
        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
    
    return test_pred
dataloaders = {'train':train_loader,"val":test_loader}
dataset_sizes = {'train':len(train_dataset),'val':len(test_dataset)}
model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,num_epochs=20)
torch.save(model,'vgg19-10.pth')
torch.save(model.state_dict(),'vgg19-10.pt')