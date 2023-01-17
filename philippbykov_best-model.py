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



# Any results you write to the current directory are saved as output.
import torch

import torch.nn.functional as F

from torch import nn, optim

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, models

import matplotlib.pyplot as plt





import pandas as pd

import numpy as np





import os

print(os.listdir("../input"))



from torch.utils.data import DataLoader
# Checking GPU is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('Training on CPU...')

else:

    print('Training on GPU...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
import torchvision.models as models

class DatasetMNIST(torch.utils.data.Dataset):

    def __init__(self, data, transform=None):

        self.data = data

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self,index):

        item = self.data.iloc[index] #обращаемся к строке по индексу

        

        image = item[1:].values.astype(np.uint8).reshape((28, 28))

        label = item[0]

        

        if self.transform is not None:

            image = self.transform(image)

            

        return image, label
BATCH_SIZE = 100

VALID_SIZE = 0.15



transform_train = transforms.Compose([transforms.ToPILImage(),

   # transforms.RandomRotation(0, 0.5),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



transform_valid = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])
dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')



train_data = DatasetMNIST(dataset, transform = transform_train)

valid_data = DatasetMNIST(dataset, transform = transform_valid)



num_train = len(train_data)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(VALID_SIZE * num_train))

train_idx, valid_idx = indices[split:] , indices[:split]



train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, sampler = train_sampler)

valid_loader = DataLoader(valid_data, batch_size = BATCH_SIZE, sampler = valid_sampler)



print(f"Lenght train: {len(train_idx)}")

print(f'Lenght validation: {len(valid_idx)}')
# Viewing data examples used for training

fig, axis = plt.subplots(3, 10, figsize=(15, 10))

images, labels = next(iter(train_loader))



for i, ax in enumerate(axis.flat):

    with torch.no_grad():

        image, label = images[i], labels[i]



        ax.imshow(image.view(28, 28), cmap='binary') # add image

        ax.set(title = f"{label}") # add label
# Viewing data examples used for validation

fig, axis = plt.subplots(3, 10, figsize=(15, 10))

images, labels = next(iter(valid_loader))



for i, ax in enumerate(axis.flat):

    with torch.no_grad():

        image, label = images[i], labels[i]



        ax.imshow(image.view(28, 28), cmap='binary') # add image

        ax.set(title = f"{label}") # add label
import torch.nn as nn

import torch.utils.model_zoo as model_zoo





__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',

           'resnet152']





model_urls = {

    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',

    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',

    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

}





def conv3x3(in_planes, out_planes, stride=1):

    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,

                     padding=1, bias=False)





def conv1x1(in_planes, out_planes, stride=1):

    """1x1 convolution"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)





class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)



        return out





class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)

        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)



        return out



class ResNet(nn.Module):



    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):

        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,

                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)



        # Zero-initialize the last BN in each residual branch,

        # so that the residual branch starts with zeros, and each residual block behaves like an identity.

        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        if zero_init_residual:

            for m in self.modules():

                if isinstance(m, Bottleneck):

                    nn.init.constant_(m.bn3.weight, 0)

                elif isinstance(m, BasicBlock):

                    nn.init.constant_(m.bn2.weight, 0)



    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                conv1x1(self.inplanes, planes * block.expansion, stride),

                nn.BatchNorm2d(planes * block.expansion),

            )



        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):

            layers.append(block(self.inplanes, planes))



        return nn.Sequential(*layers)



    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)



        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)



        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)



        return x





def resnet18(pretrained=False, **kwargs):

    """Constructs a ResNet-18 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:

        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model
model = resnet18()

model
class MnistResNet(ResNet):

    def __init__(self):

        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)

        self.conv1 = torch.nn.Conv2d(1, 64, 

            kernel_size=(7, 7), 

            stride=(2, 2), 

            padding=(3, 3), bias=False)

        

    def forward(self, x):

        return torch.softmax(

            super(MnistResNet, self).forward(x), dim=-1)
class lol(nn.Module):

    def __init__(self):

        super(lol, self).__init__()

        self.conv1 = nn.Conv2d(1,20,5)

        self.conv2 = nn.Conv2d(20,20,5)

    def forward(self,x):

        x = F.relu(self.conv1(x))

        return F.relu(self.conv2(x))

    
class Net(nn.Module):

    def __init__(self):

        super(Net,self).__init__()

        self.conv1 = nn.Sequential(

            nn.Conv2d(1,32,3,padding = 1),

            nn.ReLU(),

            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, stride=2, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(32),

            nn.MaxPool2d(2,2),

            nn.Dropout(0.25)

        )

        

        self.conv2 = nn.Sequential(

            nn.Conv2d(32, 64, 3, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(64),

            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)

        )

        

        self.conv3 = nn.Sequential(

            nn.Conv2d(64, 128, 3, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(128),

            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)

        )

        

        self.fc = nn.Sequential(

            nn.Linear(128,10)

        )

        

    def forward(self,x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        

        x = x.view(x.size(0), -1)

        return self.fc(x)
model = MnistResNet()

if train_on_gpu:

    model.cuda()
model = Net()

print(model)



if train_on_gpu:

    model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001 , momentum = 0.9)

criterion = torch.nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, 

                                           patience=2, verbose=False, threshold=0.0001,

                                           threshold_mode='rel',cooldown=0, min_lr=0, eps=1e-08)

epochs = 25



valid_loss_min = np.Inf

train_losses, valid_losses = [], []

history_accuracy = []



for e in range(1, epochs+1):

    running_loss = 0

    

    for images, labels in train_loader:

        if train_on_gpu:

            images, labels = images.cuda(), labels.cuda()

        

        optimizer.zero_grad()

        

        ps = model(images)

        

        loss = criterion(ps, labels)

        

        loss.backward()

        

        optimizer.step()

        

        running_loss += loss.item()

    

    else:

        valid_loss = 0

        accuracy = 0

        

        with torch.no_grad():

            model.eval()

            for images, labels in valid_loader:

                if train_on_gpu:

                    images, labels = images.cuda(), labels.cuda()

                ps = model(images)

                

                _, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                valid_loss += criterion(ps, labels)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

                scheduler.step(valid_loss)

        model.train()

        

        train_losses.append(running_loss/len(train_loader))

        valid_losses.append(valid_loss/len(valid_loader))

        history_accuracy.append(accuracy/len(valid_loader))

        

        #true or not

        network_learned = valid_loss < valid_loss_min

        

        if e == 1 or e % 5 == 0 or network_learned:

            print(f'Epoch: {e}/{epochs}.. ',

                  f'Training loss: {running_loss/len(train_loader):.3f}.. ',

                  f'Validation loss: {valid_loss/len(valid_loader):.3f}.. ',

                  f'Accuracy: {accuracy/len(valid_loader):.3f}')

        if network_learned:

            valid_loss_min = valid_loss

            torch.save(model.state_dict(), 'model_stat.pt')

            print('Detected network improvement, saving current model')
%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



plt.plot(train_losses, label = 'Training Loss')

plt.plot(valid_losses, label = 'Validation Loss')

plt.legend(frameon = False)
plt.plot(history_accuracy, label = 'Validation Accuracy')

plt.legend(frameon = False)
model.load_state_dict(torch.load('model_stat.pt'))
# specify the image classes

classes = ['0', '1', '2', '3', '4',

           '5', '6', '7', '8', '9']



# track test loss

test_loss = 0.0

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



model.eval()

# iterate over test data

for data, target in valid_loader:

    # move tensors to GPU if CUDA is available

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    # calculate the batch loss

    loss = criterion(output, target)

    # update test loss 

    test_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)    

    # compare predictions to true label

    correct_tensor = pred.eq(target.data.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class

    for i in range(BATCH_SIZE):

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1



# average test loss

test_loss = test_loss/len(valid_loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(10):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %0.4f%% (%2d/%2d)' % (

            classes[i], class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2.2f%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
class DatasetSubmissionMNIST(torch.utils.data.Dataset):

    def __init__(self, file_path, transform=None):

        self.data = pd.read_csv(file_path)

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        image = self.data.iloc[index].values.astype(np.uint8).reshape((28, 28, 1))



        

        if self.transform is not None:

            image = self.transform(image)

            

        return image
transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



submissionset = DatasetSubmissionMNIST('../input/digit-recognizer/test.csv', transform=transform)

submissionloader = torch.utils.data.DataLoader(submissionset, batch_size=BATCH_SIZE, shuffle=False)
submission = [['ImageId', 'Label']]



with torch.no_grad():

    model.eval()

    image_id = 1



    for images in submissionloader:

        if train_on_gpu:

            images = images.cuda()

        log_ps = model(images)

        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(1, dim=1)

        

        for prediction in top_class:

            submission.append([image_id, prediction.item()])

            image_id += 1

            

print(len(submission) - 1)
import csv



with open('submission.csv', 'w') as submissionFile:

    writer = csv.writer(submissionFile)

    writer.writerows(submission)

    

print('Submission Complete!')