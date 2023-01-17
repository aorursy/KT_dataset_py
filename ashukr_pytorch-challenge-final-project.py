# Imports here
import torch
from torchvision import datasets,transforms
!ls ../input/pytorch_challenge/pytorch_challenge-master/flower_data/
data_dir = '../input/pytorch_challenge/pytorch_challenge-master/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
batch_size = 32
# TODO: Define your transforms for the training and validation sets
train_transforms =transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) 
valid_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir,train_transforms)
valid_data = datasets.ImageFolder(valid_dir,valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_data,batch_size = batch_size,shuffle = True)
!ls ../input/pytorch_challenge/pytorch_challenge-master/cat_to_name.json
import json

with open('../input/pytorch_challenge/pytorch_challenge-master/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
cat_to_name
cat_to_name['1']
import matplotlib.pyplot as plt
import numpy as np
images,labels = next(iter(train_loader))
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(cat_to_name[str(labels[idx].item())])
# TODO: Build and train your network
from torch import nn
from torchvision import models
model = models.densenet121(pretrained=True)
print(model)
#time to freeze the gradients
for params in model.parameters():
    params.require_gad = False
import torch.nn.functional as F 
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                ("fc1",nn.Linear(1024,180)),
                ("relu",nn.ReLU()),
                ("dropout",nn.Dropout(p=0.02)),
                ("fc2",nn.Linear(180,120)),
                ("relu2",nn.ReLU()),
                ("dropout",nn.Dropout(p=0.02)),
                ("fc3",nn.Linear(120,102)),
                ("relu3",nn.ReLU())
]))
criterion = nn.CrossEntropyLoss()
model.classifier = classifier
optimizer = torch.optim.SGD(classifier.parameters(),lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
epoch = 30
min_val = np.inf
model.cuda()
for e in range(epoch):
    scheduler.step()
    train_loss = 0
    valid_loss = 0
    for images,labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)
    model.eval()
    for images, labels in valid_loader:
        images = images.cuda()
        labels = labels.cuda()
        pred = model(images)
        loss = criterion(pred,labels)
        valid_loss += loss.item()*images.size(0)
    
    model.train()
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    
    print("epoch {}/{}".format(e+1,epoch),
         "the train loss {:.3f}".format(train_loss),
         "the valid loss {:.3f}".format(valid_loss))
    
    if valid_loss <=min_val:
        torch.save(model.state_dict,'model.pth')
        print("saved the model")
        min_val = valid_loss