class_path = "/kaggle/input/cgiar-computer-vision-for-crop-disease/train/train/healthy_wheat"
! ls {class_path}
class_path = "/kaggle/input/cgiar-computer-vision-for-crop-disease/train/train/healthy_wheat"
! ls {class_path}
import matplotlib.pyplot as plt

image = plt.imread(class_path+"/I9JOL9.jpg")
plt.imshow(image)
plt.show()
# Images are matrix
image.shape
#each pixel is a color RGB
image_R = image.copy()
image_R[:,:,1]=0
image_R[:,:,2]=0
plt.imshow(image_R)
plt.show()

image_G = image.copy()
image_G[:,:,0]=0
image_G[:,:,2]=0
plt.imshow(image_G)
plt.show()

image_B = image.copy()
image_B[:,:,1]=0
image_B[:,:,0]=0
plt.imshow(image_B)
plt.show()
# each pixel contains the intensity encoded as a number from 0 to 255

image_gray = image.mean(axis=2)

plt.imshow(image_gray, cmap="gray")
plt.show()

# we can get a subimage by indexing the axis
image_small = image_gray[::50,::50]

plt.imshow(image_small, cmap="gray")
plt.show()

# for each cell, we can also check the intensity value
image_small[:4,:4]
!ls /kaggle/input/cgiar-computer-vision-for-crop-disease/train/train
import torchvision
import torch

base_dataset = torchvision.datasets.ImageFolder(
    root='/kaggle/input/cgiar-computer-vision-for-crop-disease/train/train',
    transform=None)

print(base_dataset.classes)
print(len(base_dataset))
base_dataset.imgs
image, label = base_dataset[0]

plt.imshow(image)
plt.show()
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.RandomRotation(20),
     transforms.RandomVerticalFlip()])#,
     #transforms.ToTensor(),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
for i in range(5):
    plt.imshow(transform(image))
    plt.show()
# transforms
transform = transforms.Compose(
    [transforms.Resize((224, 224)),#(224, 224)),
     transforms.RandomRotation(20),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# dataset
base_dataset = torchvision.datasets.ImageFolder(
    root='/kaggle/input/cgiar-computer-vision-for-crop-disease/train/train',
    transform=transform)

# dataloader
base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=16,shuffle=True, num_workers=4)
# where does the model do all the operations?
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# creating a model based on a preexisting architecture
from torchvision import models
import torch.nn as nn

model = models.resnet18(pretrained=False)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(base_dataset.classes))

model.to(device)
model
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# train
for epoch in range(1):
    for i, data in enumerate(base_loader, 0):
        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        loss.backward()
        optimizer.step()
import time

# create arrays to save metrics
t_start = time.time()
loss_arr=[]
acc_arr=[]
for epoch in range(1):  # loop over the dataset multiple times
    
    #initialize epoch metrics
    t_epoch = time.time()
    running_loss = 0.0
    running_acc = 0.0
    running_n = 0
    
    for i, data in enumerate(base_loader, 0):
        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        loss.backward()
        optimizer.step()
        
        # compute minibatch metrics
        running_acc += (torch.argmax(outputs, dim=1).to(device) == labels.to(device)).float().sum()
        running_n += len(labels)
        running_loss += loss.item()
        
    # compute epoch metrics
    running_acc = running_acc/running_n
    running_loss = running_loss/running_n
    loss_arr.append(running_loss)
    acc_arr.append(running_acc)
    
    # print epoch metrics
    print('epoch: {} loss: {} acc: {} time epoch: {} time total: {}'.format(epoch + 1, running_loss, running_acc, time.time()-t_epoch, time.time()-t_start))

print('Finished Training in {} seconds'.format(time.time()-t_start))
view_dataset = torchvision.datasets.ImageFolder(
    root='/kaggle/input/cgiar-computer-vision-for-crop-disease/train/train',
    transform=None)

transform_view = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

import numpy as np
indexes = np.random.choice(len(view_dataset), 5)
for i in indexes:
    x,y = view_dataset[i]
    x_view = transform_view(x)
    x_tr = transform(x)
    
    y_out = model(x_tr.to(device).unsqueeze(0))
    y_prob = nn.functional.softmax(y_out, dim=1)
    y_pred = torch.argmax(y_prob, dim=1)
    
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(x)
    axs[0].set_title("[{}] label {}".format(i,y))
    
    axs[1].imshow(transforms.ToPILImage()(x_tr))
    axs[1].set_title("pred {}".format(y_pred[0]))
    plt.show()
len(base_dataset)
# split our dataset
train_dataset, val_dataset = torch.utils.data.random_split(base_dataset, [450, 112])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16,shuffle=True, num_workers=4)
# create arrays to save metrics
t_start = time.time()
loss_arr=[]
acc_arr=[]
val_acc_arr=[]
for epoch in range(50):  # loop over the dataset multiple times
    
    #initialize epoch metrics
    t_epoch = time.time()
    running_loss = 0.0
    running_acc = 0.0
    running_n = 0
    running_val_acc = 0.0
    running_val_n = 0
    
    # train
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        loss.backward()
        optimizer.step()
        
        # compute minibatch metrics
        running_acc += (torch.argmax(outputs, dim=1).to(device) == labels.to(device)).float().sum()
        running_n += len(labels)
        running_loss += loss.item()
    
    # validate/evaluate
    for i, data in enumerate(val_loader, 0):
        # get the inputs
        inputs, labels = data

        # forward
        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        
        # compute minibatch metrics
        running_val_acc += (torch.argmax(outputs, dim=1).to(device) == labels.to(device)).float().sum()
        running_val_n += len(labels)
        
    # compute epoch metrics
    running_acc = running_acc/running_n
    running_val_acc = running_val_acc/running_val_n
    running_loss = running_loss/running_n
    loss_arr.append(running_loss)
    acc_arr.append(running_acc)
    val_acc_arr.append(running_val_acc)
    
    # print epoch metrics
    print('epoch: {} loss: {} acc: {} valacc: {} time epoch: {} time total: {}'.format(epoch + 1, running_loss, running_acc, running_val_acc, time.time()-t_epoch, time.time()-t_start))

print('Finished Training in {} seconds'.format(time.time()-t_start))
plt.plot(val_acc_arr, label="val_acc")
plt.plot(acc_arr, label="acc")
plt.legend()
plt.show()
plt.plot(loss_arr, label="loss")
plt.legend()
plt.show()
