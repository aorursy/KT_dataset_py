# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import torch

import torch.nn as nn

import torch.nn.functional as F

from PIL import Image

import matplotlib.pyplot as plt

import time

from sklearn.metrics import accuracy_score

import seaborn as sns





import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
train_path_negative = '/kaggle/input/lsifir/LSIFIR/Classification/Train/neg'

train_path_positive = '/kaggle/input/lsifir/LSIFIR/Classification/Train/pos'

train = [train_path_negative, train_path_positive]



test_path_negative = '/kaggle/input/lsifir/LSIFIR/Classification/Test/neg'

test_path_positive = '/kaggle/input/lsifir/LSIFIR/Classification/Test/pos'

test = [test_path_negative, test_path_positive]





def read_images(path):

    x = []

    y = []

    label = 0

    i = 0

    for folder in path:

        for file in os.listdir(folder):

            im = Image.open(os.path.join(folder, file))

            x.append(np.asarray(im, dtype='uint8'))

            y.append(label)



            i += 1

            if i % 10000 == 0:

                print('Number of samples read: {}'.format(i))

        label += 1



    return torch.tensor(x), torch.tensor(y)







def show_images(x, y, number_of_samples):

    

    classes = len(np.unique(y))



    fig, ax = plt.subplots(classes, number_of_samples, figsize=(15,7))

    

    for i in range(classes):

        indices = np.where(y == i)[0]

        for j in range(number_of_samples):

            ax[i][j].imshow(x[np.random.choice(indices)], cmap='gray')

            ax[i][j].set_title('Non-Pedestrian' if i ==0 else 'Pedestrian')

            ax[i][j].axis('off')

            

    plt.show()
x_train, y_train = read_images(train)

x_test, y_test = read_images(test)
print('Number of Train Images: Positive {}, Negative {}\nNumber of Test Images: Positive {}, Negative {}'

      .format(len(os.listdir(train_path_positive)), len(os.listdir(train_path_negative)),

              len(os.listdir(test_path_positive)), len(os.listdir(test_path_negative))))



show_images(x_train, y_train, 5)
classes = np.unique(y_train).size

learning_rate = 0.00001





class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

                               #color channel, # of conv layers

        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 10, kernel_size= 3)

        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        self.conv2 = nn.Conv2d(10, 20, 3)

        self.neurons = self.linear_input_neurons()

        

        

        self.fc1 = nn.Linear(self.linear_input_neurons(), 500)

        self.fc2 = nn.Linear(500, 250)

        self.fc3 = nn.Linear(250, classes)



    def forward(self, x):

        x = self.maxpool(F.relu(self.conv1(x.float())))

        x = self.maxpool(F.relu(self.conv2(x.float())))

        x = x.view(-1, self.neurons)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        

        return x

    

    def size_after_relu(self, x):

        x = self.maxpool(F.relu(self.conv1(x.float())))

        x = self.maxpool(F.relu(self.conv2(x.float())))

        

        return x.size()

    

    def linear_input_neurons(self):

        size = self.size_after_relu(torch.rand(1, 1, 64, 32))

        m = 1

        for i in size:

            m *= i



        return int(m)

    

    

device = torch.device('cuda' if torch.cuda.is_available() == True else 'cpu')

model = CNN().to(device)
batch_size = x_train.size(0)//5



import torch.utils.data

train = torch.utils.data.TensorDataset(x_train, y_train)

trainloader = torch.utils.data.DataLoader(train, batch_size= batch_size, shuffle= True) # ( tensor(images), tensor(labels) )



test = torch.utils.data.TensorDataset(x_test, y_test)

testloader = torch.utils.data.DataLoader(test, batch_size= batch_size, shuffle= False)
criterion = nn.CrossEntropyLoss()



import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.8)
train_acc1 = []

test_acc1  = []

loss_list1 = []

iterations = []

epochs = 500





for epoch in range(epochs+1):

    for i, data in enumerate(trainloader):

        

        images, labels = data

        images = images.view(images.size(0), 1, 64, 32)

        

        # gpu or cpu

        images, labels = images.to(device), labels.to(device)

        

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        

    # test accuracy

    correct = 0

    total = 0

    with torch.no_grad():

        for data in testloader:

            images, labels = data

            images = images.view(images.size(0), 1, 64, 32)

        

            images, labels_test = images.to(device), labels.to(device)

            

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels_test.size(0)

            correct += (predicted == labels_test).sum().item()

    

    accuracy1 = correct / total



    

    # train accuracy

    correct = 0

    total = 0

    with torch.no_grad():

        for data in trainloader:

            images, labels = data

            images = images.view(images.size(0), 1, 64, 32)

        

            images, labels_train = images.to(device), labels.to(device)

            

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels_train.size(0)

            correct += (predicted == labels_train).sum().item()

    

    accuracy2 = correct / total

    

    if epoch % 25 == 0: 

        loss_list1.append(loss.item())

        train_acc1.append(accuracy2)

        test_acc1.append(accuracy1)

        iterations.append(epoch)

        

        print('epoch: {}   -->  train accuracy = {:.5f},\ttest accuracy = {:.5f},\tloss = {:.5f}'.format(epoch, accuracy2, accuracy1, loss.item()))

    



print('Finished.')
fig, ax1 = plt.subplots(figsize=(12,7))



ax1.plot(np.array(iterations)+1, test_acc1, label='Test Accuracy', marker='o', color='green')

ax1.plot(np.array(iterations)+1, train_acc1, label='Train Accuracy', marker='s', color='blue')

ax1.set_xlabel('Epoch', fontsize=13)

ax1.set_ylabel('Accuracy', fontsize=13)

plt.legend()



ax2 = ax1.twinx()

ax2.plot(np.array(iterations)+1, loss_list1, label='Loss', marker='P', color='red')

ax2.set_ylabel('Loss', fontsize=13)

plt.legend()

plt.title('Accuracies & Loss', fontsize=20)

plt.show()
class_correct = list(0. for i in range(2))

class_total = list(0. for i in range(2))

with torch.no_grad():

    for data in testloader:

        

        images,labels = data

        images = images.view(images.size(0), 1, 64, 32)

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        _,predicted = torch.max(outputs, 1)

        c = (predicted == labels).squeeze()

        for i in range(labels.size()[0]):

            label = labels[i]

            class_correct[label] += c[i].item()

            class_total[label] += 1



classess = ['Pedestrians(0)','Non-Pedestrians(1)']

for i in range(2):

    print('Accuracy of %s : %2d %%' % (

        classess[i], 100 * class_correct[i] / class_total[i]))

    

    

c = np.concatenate((np.array(class_correct).reshape(-1,1), np.array(class_total).reshape(-1,1) - np.array(class_correct).reshape(-1,1) ), axis= 1)

plt.figure(figsize= (10,7))

sns.heatmap(c, annot= True, linewidths=0.5, fmt='g')

plt.show()
classes = np.unique(y_train).size

learning_rate = 0.00001





# block layers

def conv3x3(in_planes, out_planes, stride=1):            # stride=3 & padding=1 --> size remains the same!

    return nn.Conv2d(in_planes, out_planes, kernel_size= 3, stride= stride, padding=1, bias=False) # bias = False --> Batch Normalization already includes the bias term.



# downsampling

def conv1x1(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size= 1, stride= stride, bias= False)







class BasicBlock(nn.Module):

    

    expansion = 1

    

    def __init__(self,inplanes, planes, stride = 1, downsample = None):

        super(BasicBlock,self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace = True)

        self.drop = nn.Dropout(0.9)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.stride = stride

        

    def forward(self, x):

        identity = x

        

        out = self.drop(self.relu(self.bn1(self.conv1(x.float()))))

        out = self.drop(self.bn2(self.conv2(out.float())))

        

        if self.downsample is not None:

            identity = self.downsample(x)

            

        out += identity

        out = self.relu(out)

        return out

    

    



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes = classes):

        super(ResNet,self).__init__()

        

        # before block layers

        self.inplanes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride = 2, padding = 3, bias= False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace = True)

        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride = 2, padding = 1)

        

        # block layers

        self.layer1 = self._make_layer(block, 64,  layers[0], stride = 1)

        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)

    

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # target output size: (1,1) after adaptive pooling

        self.fc = nn.Linear(256*block.expansion, num_classes)

    

        

        # initializing weights

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

    

    

    def _make_layer(self, block, planes, blocks, stride):

        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion:   # size changes after convolution operations

            # create downsample method

            downsample = nn.Sequential(

                    conv1x1(self.inplanes, planes*block.expansion, stride),

                    nn.BatchNorm2d(planes*block.expansion))

            

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample)) # send 

        self.inplanes = planes*block.expansion # conv1's output dimension == conv2's input dimension

        

        for _ in range(1,blocks):

            layers.append(block(self.inplanes, planes)) # merge blocks

        

        return nn.Sequential(*layers)

        

    def forward(self, x):

        x = self.maxpool(self.relu(self.bn1(self.conv1(x.float()))))

        x = self.layer1(x.float())

        x = self.layer2(x.float())

        x = self.layer3(x.float())

        x = self.avgpool(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        

        return x
batch_size = x_train.size(0)//5



import torch.utils.data

train = torch.utils.data.TensorDataset(x_train, y_train)

trainloader = torch.utils.data.DataLoader(train, batch_size= batch_size, shuffle= True) # ( tensor(images), tensor(labels) )



test = torch.utils.data.TensorDataset(x_test, y_test)

testloader = torch.utils.data.DataLoader(test, batch_size= batch_size, shuffle= False)
device = torch.device('cuda' if torch.cuda.is_available() == True else 'cpu')

model = ResNet(BasicBlock, [2,2,2]).to(device)



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
train_acc2 = []

test_acc2  = []

loss_list2 = []

iterations = []



epochs = 500





for epoch in range(epochs+1):

    for (images,labels) in trainloader:

        

        images = images.view(images.size(0), 1, 64, 32)

        images, labels = images.to(device), labels.to(device)

        

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        

    # test accuracy

    correct = 0

    total = 0

    with torch.no_grad():

        for (images, labels) in testloader:

            

            images = images.view(images.size(0), 1, 64, 32)

            images, labels = images.to(device), labels.to(device)

            

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)  # return max of index(1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    

    accuracy1 = correct / total



    

    # train accuracy

    correct = 0

    total = 0

    with torch.no_grad():

        for (images, labels) in trainloader:

            

            images = images.view(images.size(0), 1, 64, 32)

            images, labels = images.to(device), labels.to(device)

            

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    

    accuracy2 = correct / total

    if epoch % 25 == 0:

        test_acc2.append(accuracy1)

        train_acc2.append(accuracy2)

        loss_list2.append(loss.item())

        iterations.append(epoch)

        

        print('epoch: {}   -->  train accuracy = {:.5f},\ttest accuracy = {:.5f},\tloss = {:.5f}'.format(epoch, accuracy2, accuracy1, loss.item()))

    



print('Finished.')
fig, ax1 = plt.subplots(figsize=(12,7))



ax1.plot(np.array(iterations)+1, test_acc2, label='Test Accuracy', marker='o', color='green')

ax1.plot(np.array(iterations)+1, train_acc2, label='Train Accuracy', marker='s', color='blue')

ax1.set_xlabel('Epoch', fontsize=13)

ax1.set_ylabel('Accuracy', fontsize=13)

plt.legend()



ax2 = ax1.twinx()

ax2.plot(np.array(iterations)+1, loss_list2, label='Loss', marker='P', color='red')

ax2.set_ylabel('Loss', fontsize=13)

plt.legend()

plt.title('Accuracies & Loss', fontsize=20)

plt.show()
class_correct = list(0. for i in range(2))

class_total = list(0. for i in range(2))

with torch.no_grad():

    for data in testloader:

        

        images,labels = data

        images = images.view(images.size(0), 1, 64, 32)

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        _,predicted = torch.max(outputs, 1)

        c = (predicted == labels).squeeze()

        for i in range(labels.size()[0]):

            label = labels[i]

            class_correct[label] += c[i].item()

            class_total[label] += 1



classess = ['Pedestrians(0)','Non-Pedestrians(1)']

for i in range(2):

    print('Accuracy of %s : %2d %%' % (

        classess[i], 100 * class_correct[i] / class_total[i]))

    

    

c = np.concatenate((np.array(class_correct).reshape(-1,1), np.array(class_total).reshape(-1,1) - np.array(class_correct).reshape(-1,1) ), axis= 1)

plt.figure(figsize= (10,7))

sns.heatmap(c, annot= True, linewidths=0.5, fmt='g')

plt.show()
d = pd.DataFrame({'model':['CNN' for _ in range(3)] + ['DRN' for _ in range(3)],

                  'phase':(['Train Accuracy' for _ in range(1)] + ['Test Accuracy' for _ in range(1)] + ['Train Loss' for _ in range(1)])*2,

                  'accs':[train_acc1[-1],test_acc1[-1],loss_list1[-1], train_acc2[-1],test_acc2[-1],loss_list2[-1]]})



hm = d.pivot('model', 'phase', 'accs')





plt.figure(figsize=(15,9))

sns.heatmap(hm, annot= True, linewidths= 0.5, annot_kws= {'size':12, 'weight':'bold'})

plt.yticks(rotation= 0, fontsize= 12)

plt.xticks(fontsize= 12)

plt.xlabel('Results', fontsize= 18)

plt.ylabel('Model', fontsize= 18, rotation= 0)

plt.show()