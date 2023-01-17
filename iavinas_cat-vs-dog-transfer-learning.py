import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import PIL
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
use_cuda = torch.cuda.is_available()

!ls ../input/cat-and-dog
!ls ../input/cat-and-dog/training_set/training_set/
!ls ../input/cat-and-dog/test_set/test_set

batch_size = 128
num_workers = 2
data_transform = { 
    'train' : torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
    'test': torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])}

train_dir = '../input/cat-and-dog/training_set/training_set/'

test_dir = '../input/cat-and-dog/test_set/test_set'
train_data = datasets.ImageFolder(train_dir, transform = data_transform['train'])
test_data = datasets.ImageFolder(test_dir, transform = data_transform['test'])
print('Num training images: ', len(train_data))

print('Num test images: ', len(test_data))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle= True)

loader = {'train': train_loader, 'val' : test_loader}
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()
classes = ['cat', 'dog']
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(10):
    ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])
model = models.resnet50(pretrained= True)
for param in model.parameters():
    param.requires_grad = False
print(model)
fc_inputs = model.fc.in_features
 
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 1),
    
)

print(model)
if use_cuda:
    model = model.cuda()
import torch.optim as optim


criterion = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model.fc.parameters(),lr=0.01, momentum=0.9)
valid_loss_min = np.Inf
n_classes = 1
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):

    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        #print(use_cuda)
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            #print(output.shape, target.shape)
            loss = criterion(output, target.view(-1,1).float())
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['val']):
            
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target.view(-1, 1).float())
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
    return model


# train the model
model_trained = train(20, loader, model, optimizer, 
                      criterion, use_cuda, 'model_transer.pt')
model_trained.load_state_dict(torch.load('model_transer.pt'))
def test(loaders, model, criterion, use_cuda):
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['val']):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        
        loss = criterion(output, target.view(-1,1).float())
        
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        
        pred = (output>0).float()
        
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
   
test(loader, model_trained, criterion, use_cuda)
