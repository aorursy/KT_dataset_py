# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
#!pip install 'tensorflow==1.15.0'
#!pip install 'tensorflow-gpu==1.15'
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data.sampler import SubsetRandomSampler
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass
#torch.cuda.set_device(0)                
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_transforms_light = {
    'train': transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(51),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(51),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_transforms_32 = {
    'train': transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = '/kaggle/input'
train_path = "/kaggle/input/dermoscopy-images/train/train"
val_path = "/kaggle/input/dermoscopy-images/val/val"

train_set = datasets.ImageFolder((train_path), data_transforms_32["train"])
val_set = datasets.ImageFolder((val_path), data_transforms_32["val"])
print("nice")


batch_size = 32
validation_split = .5
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(val_set)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
test_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=4)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                sampler=test_sampler, num_workers=4)




#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
#                                             shuffle=True, num_workers=4)
#              for x in ['train', 'val']}
#dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#class_names = image_datasets['train'].classes

device = (1 if torch.cuda.is_available() else 2)
print("device: ", device)

## MODEL #############################################################################################################################################################################

def resnet18_pretrained():
    model = torchvision.models.resnet18(pretrained=True)
    model = model.float()
    if(device == 1):
        model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    if(device == 1):
        model.fc = nn.Linear(num_ftrs, 2).cuda()
    else:
        model.fc = nn.Linear(num_ftrs, 2)


    if(device == 1):
        model = model.cuda()

    from torch import optim
    # Loss and optimizer
    criteration = nn.CrossEntropyLoss() #criteration = nn.NLLLoss()
    criteration = criteration
    optimizer = optim.Adam(model.fc.parameters())
    return model, criteration, optimizer

def alexnet_pretrained():
    if(device == 1):
        use_gpu = True
    else:
        use_gpu = False
    
    model_ft = models.alexnet(pretrained=True)

    # create two list for feature and classifier blocks
    ft_list = list(model_ft.features)
    cl_list = list(model_ft.classifier)

    # Modifing feature block after removing the last max-pool layer (ft_list[12])
    model_ft.features = nn.Sequential(ft_list[0], ft_list[1], ft_list[2],
                                      ft_list[3], ft_list[4], ft_list[5],
                                      ft_list[6], ft_list[7], ft_list[8],
                                      ft_list[9], ft_list[10], ft_list[11])

    num_ftrs_out = 256
    num_ftrs = 2 * 2 * num_ftrs_out

    # Modifing classifier block with a dropout and a fc layer only
    cl_list[1] = nn.Linear(num_ftrs, 2)
    model_ft.classifier = nn.Sequential(cl_list[0], cl_list[1])

    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    return model_ft, criterion, optimizer_ft

def vgg16_pretrained():
    use_gpu = False
    if(device == 1):
        use_gpu = True
    
    vgg16 = models.vgg16_bn()
    #vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    print(vgg16.classifier[6].out_features) # 1000 


    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, 2)]) # Add our layer with 4 outputs
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
    if use_gpu:
        vgg16.cuda() #.cuda() will move everything to the GPU side
    
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return vgg16, criterion, optimizer_ft

import torch.nn.functional as F

def custom_0():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5) # 224 -> 220
            self.pool1 = nn.MaxPool2d(2, 2) # 110
            self.conv2 = nn.Conv2d(6, 16, 5) # 106
            self.pool2 = nn.MaxPool2d(2, 2) # 53
            self.fc1 = nn.Linear(5*5* 16, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 2)

        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            #print("x shape: ", x.shape)
            x = x.view(-1, 5*5 * 16)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    net = Net()
    if(device == 1):
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9)
    return net, criterion, optimizer

def custom_1():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5) # 224 -> 220
            self.pool1 = nn.MaxPool2d(2, 2) # 110
            self.conv2 = nn.Conv2d(6, 16, 5) # 106
            self.pool2 = nn.MaxPool2d(2, 2) # 53
            self.conv3 = nn.Conv2d(16, 16, 4) # 50
            self.pool3 = nn.MaxPool2d(2, 2) # 25
            self.conv4 = nn.Conv2d(16, 10, 4) # 22
            self.pool4 = nn.MaxPool2d(2, 2) # 11
            self.fc1 = nn.Linear(11*11* 10, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc21 = nn.Linear(84, 42)
            self.fc3 = nn.Linear(42, 2)

        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.pool4(F.relu(self.conv4(x)))
            #print("x shape: ", x.shape)
            x = x.view(-1, 11*11* 10)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc21(x))
            x = self.fc3(x)
            return x
    net = Net()
    if(device == 1):
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9)
    return net, criterion, optimizer

def squeezenet():
    model = torchvision.models.squeezenet1_0(pretrained=False)
    model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1E-3, momentum=0.9)
    print("/",device,"/")
    if(device == 1):
        print("CUDA")
        model.cuda()
    return model, criterion, optimizer
model, criterion, optimizer = custom_0()
## TRAIN #############################################################################################################################################################################
import time
timer = time.time()
print("device: ", device)

optimizer.zero_grad()
#all_preds = torch.tensor([])

def increment_conf_matrix(predicted, labels, confusion_matrix):
    #print("hm: ", (labels), (predicted))
    for i in range(len(predicted)):
      pred = int(predicted[i]) # 0
      true = int(labels[i]) # 1
      #[0][1]
      # 0 1
      # 0 0
      confusion_matrix[true][pred] += 1
    
def get_correct_and_total(predicted, labels):
    #print("TEST")
    total = labels.size(0) 
    #print(predicted)
    #print(labels)
    
    correct = (predicted == labels).sum()  
    #print(correct)
    #print(total)
    return correct, total
 
def accuracy_from_confusion(conf_mat):
    corrects = conf_mat[0][0] + conf_mat[1][1]
    total = conf_mat[0][0] + conf_mat[1][1] + conf_mat[0][1] + conf_mat[1][0]
    return (corrects*1.0/total*1.0)

def find_mean(arr):
    total = 0.0
    for e in arr:
        total += float(e)
    return total/len(arr)
    
#print("shape: ", confusion_matrix.shape)
def train_epoch(loader, enable_training, conf_matrix = False):
    batch_losses = []
    batch_accuracies = []
    

    if(True):
      i = 0
      running_loss = 0.0
      running_correct = 0.0
      running_total = 0.0
      for batch in loader:
            inputs = batch[0].float()
            labels = batch[1]
            if(device == 1):
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            if(enable_training):
                optimizer.zero_grad()
    
            # forward + backward + optimize
            #print("inputs shape: ",inputs.shape)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            if(enable_training):
                loss.backward()
                optimizer.step()

            # print statistics
            running_loss += loss.item()
            corrects, total = get_correct_and_total(predicted, labels)
            running_correct += corrects
            running_total += total

            
            if(conf_matrix != False):
                increment_conf_matrix(predicted, labels, conf_matrix)
            
            print_interval = 1
            if i % print_interval == print_interval-1:    # print every 2000 mini-batches
                print("loss: ", (i, running_loss / print_interval), ", acc: ", running_correct/running_total)
                batch_losses.append(running_loss / print_interval)
                if(conf_matrix != False):
                    #print("EQUAL?: ", accuracy_from_confusion(conf_matrix), (running_correct/running_total))
                    pass
                batch_accuracies.append(running_correct/running_total)
                running_loss = 0.0
                running_correct = 0.0
                running_total = 0.0
            i+=1
            
      if(conf_matrix != False):
        print("EQUAL?: ", accuracy_from_confusion(conf_matrix),  find_mean(test_accuracies))
        
      return batch_losses, batch_accuracies
          

    #print('epoch end, testing')
'''
def test_epoch():
    confusion_matrix = np.zeros((2,2))
    correct = 0
    total = 0
    np.zeros((2,2)) # reset matrix
    batchcounter = 0
    for batch in test_loader:
      #images = Variable(images.view(-1, 28*28))
      
      images = batch[0]
      labels = batch[1]
      if(device):
        images = images.cuda()
        labels = labels.cuda()
    
    
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score

      for i in range(len(predicted)):
        pred = predicted[i]
        true = labels[i]
        #print("ff: ", int(true), int(pred))
        confusion_matrix[int(true)][int(pred)] += 1

      total += labels.size(0)                    # Increment the total count
      correct += (predicted == labels).sum()     # Increment the correct count
      #print("b: ", batchcounter, "/", len(test_loader))
      batchcounter += 1
    print("Validation accuracy: ", 100 * (correct * 1.0 / total * 1.0))
    return (1.0 * correct / total * 1.0), confusion_matrix # validation accuracy
    
'''
print("a")

all_train_losses = []
all_test_losses = []
all_train_accuracies = []
all_test_accuracies = []

confusion_matrix = [[0, 0], [0, 0]]
print(confusion_matrix)
test_accuracies, test_losses = train_epoch(test_loader, False, confusion_matrix)
all_test_losses = all_test_losses + [find_mean(test_losses)]
all_test_accuracies = all_test_accuracies + [find_mean(test_accuracies)]
print(all_test_losses)
print(all_test_accuracies)


for epoch in range(2):
    train_losses, train_accuracies = train_epoch(train_loader, True)
    all_train_losses = all_train_losses + train_losses
    all_train_accuracies = all_train_accuracies + train_accuracies
    
    best_acc = accuracy_from_confusion(confusion_matrix)
    
    temp_confusion_matrix = [[0, 0], [0, 0]]
    test_accuracies, test_losses = train_epoch(test_loader, False, temp_confusion_matrix)
    
    all_test_losses = all_test_losses + [find_mean(test_losses)]
    print("before: ", all_test_accuracies)
    #all_test_accuracies = all_test_accuracies + test_accuracies
    print("after: ", all_test_accuracies)
    
    #print("check: ", accuracy_from_confusion(temp_confusion_matrix), )
    temp_acc = accuracy_from_confusion(temp_confusion_matrix)
    all_test_accuracies = all_test_accuracies + [find_mean(test_accuracies)]
    print("after: ", all_test_accuracies)
    print("comparison: ", temp_acc, ">", best_acc)
    if(temp_acc > best_acc):
        print("improved")
        confusion_matrix = [[temp_confusion_matrix[0][0], temp_confusion_matrix[0][1]], [temp_confusion_matrix[1][0], temp_confusion_matrix[1][1]]]
        best_acc = temp_acc
        print("new conf: ", confusion_matrix)
from tabulate import tabulate

table = [
    ["", "pred. mlg", "pred. nevus"],
    ["true mlg", confusion_matrix[0][0],confusion_matrix[0][1]],
    ["true nevus",confusion_matrix[1][0],confusion_matrix[1][1]]
]

print(tabulate(table))


import matplotlib.pyplot as plt 
#plt.ylim((0, 1))
print_interval = 1.0
batch_per_epoch = len(train_loader)
batch_per_epoch_test = len(test_loader)
print(test_accuracies)


x1 = [i * print_interval/batch_per_epoch for i in range(len(all_train_losses))]
y1 = all_train_losses
plt.plot(x1, y1, label = "train loss") 

x4 = [i * print_interval/1 for i in range(len(all_test_losses))] 
y4 = all_test_losses
plt.plot(x4, y4, label = "test loss") 

plt.xlabel('epoch') 
plt.ylabel('value') 
plt.title('losses') 
plt.legend() 
plt.show() 

x2 = [i * print_interval/batch_per_epoch for i in range(len(all_train_accuracies))]
y2 = all_train_accuracies
plt.plot(x2, y2, label = "train accuracy") 

x3 = [i * print_interval/1 for i in range(len(all_test_accuracies))] 
y3 = all_test_accuracies
plt.plot(x3, y3, label = "test accuracy") 


plt.xlabel('epoch') 
plt.ylabel('value') 
plt.title('accuracies') 
plt.legend() 
plt.show() 
x1 = 0
x2 = 0
x3 = 0
x4 = 0
print(all_test_losses)
