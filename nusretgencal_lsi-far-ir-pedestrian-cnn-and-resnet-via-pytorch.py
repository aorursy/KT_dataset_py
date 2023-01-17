# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.ignore("warnings")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import torch.nn as nn # neural networks

import torch.nn.functional as F

from PIL import Image # PIL library for read images

import matplotlib.pyplot as plt

import numpy as np

import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

print("Device:", device)
def read_images(path, num_img):

    array = np.zeros([num_img, 64*32]) # it looks like zero array [number_of_images, 64*32] 64*32 (2048 column)

    i = 0

    for img in os.listdir(path):

        img_path = path + "//" + img 

        img = Image.open(img_path, mode = "r")

        data = np.asarray(img, dtype = "uint8")

        data = data.flatten() # 1x2048

        array[i,:] = data # 1x2048(64*32) img to new array

        i += 1

    return array
train_negative_path = r'/kaggle/input/lsifir/LSIFIR/Classification/Train/neg'

num_train_neg_img = 43390

train_negative_array = read_images(train_negative_path, num_train_neg_img)
print("x_train_negative_array_shape:", train_negative_array.shape)

train_negative_array
test_negative_path = r'/kaggle/input/lsifir/LSIFIR/Classification/Test/neg'

num_test_neg_img = 22050

test_negative_array = read_images(test_negative_path, num_test_neg_img)
print("x_test_negative_array_shape:", test_negative_array.shape)

test_negative_array
train_positive_path = r'/kaggle/input/lsifir/LSIFIR/Classification/Train/pos'

num_train_pos_img = 10208

train_positive_array = read_images(train_positive_path, num_train_pos_img)
print("x_train_positive_array_shape:", train_positive_array.shape)

train_positive_array
test_positive_path = r'/kaggle/input/lsifir/LSIFIR/Classification/Test/pos'

num_test_pos_img = 5944

test_positive_array = read_images(test_positive_path, num_test_pos_img)
print("x_test_positive_array_shape:", test_positive_array.shape)

test_positive_array
x_train_negative_tensor = torch.from_numpy(train_negative_array) ## numpy to tensor for using torch

print("x_train_negative_tensor:", x_train_negative_tensor.size())

y_train_negative_tensor = torch.zeros(num_train_neg_img, dtype = torch.long) ## this is y_train tensor for labels

print("y_train_negative_tensor:", y_train_negative_tensor.size())
x_test_negative_tensor = torch.from_numpy(test_negative_array[:20855,:]) ## numpy to tensor for using torch

print("x_test_negative_tensor:", x_test_negative_tensor.size())

y_test_negative_tensor = torch.zeros(20855, dtype = torch.long) ## this is y_train tensor for labels

print("y_test_negative_tensor:", y_test_negative_tensor.size())
x_train_positive_tensor = torch.from_numpy(train_positive_array) ## numpy to tensor for using torch

print("x_train_positive_tensor:", x_train_positive_tensor.size())

y_train_positive_tensor = torch.ones(num_train_pos_img, dtype = torch.long) ## this is y_train tensor for labels

print("y_train_positive_tensor:", y_train_positive_tensor.size())
x_test_positive_tensor = torch.from_numpy(test_positive_array) ## numpy to tensor for using torch

print("x_test_positive_tensor:", x_test_positive_tensor.size())

y_test_positive_tensor = torch.ones(num_test_pos_img, dtype = torch.long) ## this is y_train tensor for labels

print("y_test_positive_tensor:", y_test_positive_tensor.size())
x_train = torch.cat((x_train_negative_tensor, x_train_positive_tensor), 0)

y_train = torch.cat((y_train_negative_tensor, y_train_positive_tensor), 0)

print("x_train:", x_train.size())

print("y_train:", y_train.size())
x_test = torch.cat((x_test_negative_tensor, x_test_positive_tensor), 0)

y_test = torch.cat((y_test_negative_tensor, y_test_positive_tensor), 0)

print("x_test:", x_test.size())

print("y_test:", y_test.size())
plt.imshow(x_train[45001,:].reshape(64, 32),)
plt.imshow(x_train[45001,:].reshape(64, 32), cmap = "gray")
num_epochs = 50 #5000

num_classes = 2

batch_size = 100 #8933

lr = 0.00001



class Net(nn.Module):

    

    def __init__(self):

        super(Net, self).__init__()

        

        self.conv1 = nn.Conv2d(1, 10, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(10, 16, 5)

        

        self.fc1 = nn.Linear(16*13*5, 520) # fully connected layer

        self.fc2 = nn.Linear(520, 130) 

        self.fc3 = nn.Linear(130, num_classes)

        

# # conv1 > relu > pooling > conv2 > relu > pooling > flatten > fc1 > relu > fc2 > relu > output



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x))) 

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16*13*5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

    
import torch.utils.data



train = torch.utils.data.TensorDataset(x_train, y_train)

trainloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = "True")



test = torch.utils.data.TensorDataset(x_test, y_test)

testloader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = "True")



net = Net().to(device) #for gpu/cpu
import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr = lr, momentum = 0.8)



start = time.time()

train_acc = []

test_acc = []

loss_list = []



use_gpu = True # False for cpu



for epoch in range(num_epochs): 

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        inputs = inputs.view(inputs.size(0), 1, 64, 32) # reshape

        inputs = inputs.float() # float

        

        #use gpu

        if use_gpu:

            if torch.cuda.is_available():

                inputs, labels = inputs.to(device), labels.to(device)



        #zero gradient

        optimizer.zero_grad()

        

        #forward

        outputs = net(inputs)

        

        #loss

        loss = criterion(outputs, labels)

        

        #back

        loss.backward()

        

        #update weights

        optimizer.step()

        

    #test

    

    correct = 0

    total = 0

    with torch.no_grad(): # cancel back propagation

        for data in testloader:

            images, labels = data 

            images = images.view(images.size(0), 1, 64, 32)

            images = images.float()

            

            #use gpu

            if use_gpu:

                if torch.cuda.is_available():

                    images, labels = images.to(device), labels.to(device)

            

            outputs = net(images)

            

            _, predicted = torch.max(outputs.data, 1)

            

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    

    acc1 = 100 * (correct / total)

    print("accuracy test", acc1)

    test_acc.append(acc1)

    

    #train

    

    correct = 0

    total = 0

    with torch.no_grad(): # cancel back propagation

        for data in trainloader:

            images, labels = data 

            images = images.view(images.size(0), 1, 64, 32)

            images = images.float()

            

            #use gpu

            if use_gpu:

                if torch.cuda.is_available():

                    images, labels = images.to(device), labels.to(device)

            

            outputs = net(images)

            

            _, predicted = torch.max(outputs.data, 1)

            

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    

    acc2 = 100 * (correct / total)

    print("accuracy train", acc2)

    train_acc.append(acc2)

    

print("train is done")

    

end = time.time()

process_time = (end - start) / 60

print("process time:", process_time)
fig, ax1 = plt.subplots()

plt.plot(loss_list, label = "Loss", color = "blue")

ax2 = ax1.twinx()

ax2.plot(np.array(test_acc)/100, label = "Test Acc", color = "green")

ax2.plot(np.array(train_acc)/100, label = "Train Acc", color = "red")

ax1.legend()

ax2.legend()

ax1.set_xlabel("Epoch")

fig.tight_layout()

plt.title("Loss vs Test Accuracy")

plt.show()

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.ignore("warnings")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import torch.nn as nn # neural networks

import torch.nn.functional as F

from PIL import Image # PIL library for read images

import matplotlib.pyplot as plt

import numpy as np

import time

import torch.utils.data



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

print("Device:", device)



#read images func



def read_images(path, num_img):

    array = np.zeros([num_img, 64*32]) # it looks like zero array [number_of_images, 64*32] 64*32 (2048 column)

    i = 0

    for img in os.listdir(path):

        img_path = path + "//" + img 

        img = Image.open(img_path, mode = "r")

        data = np.asarray(img, dtype = "uint8")

        data = data.flatten() # 1x2048

        array[i,:] = data # 1x2048(64*32) img to new array

        i += 1

    return array



#read train negative

train_negative_path = r'/kaggle/input/lsifir/LSIFIR/Classification/Train/neg'

num_train_neg_img = 43390

train_negative_array = read_images(train_negative_path, num_train_neg_img)

x_train_negative_tensor = torch.from_numpy(train_negative_array[:42000, :]) ## numpy to tensor for using torch

print("x_train_negative_tensor:", x_train_negative_tensor.size())

y_train_negative_tensor = torch.zeros(42000, dtype = torch.long) ## this is y_train tensor for labels

print("y_train_negative_tensor:", y_train_negative_tensor.size())



#read train positive 

train_positive_path = r'/kaggle/input/lsifir/LSIFIR/Classification/Train/pos'

num_train_pos_img = 10208

train_positive_array = read_images(train_positive_path, num_train_pos_img)

x_train_positive_tensor = torch.from_numpy(train_positive_array[:10000, :]) ## numpy to tensor for using torch

print("x_train_positive_tensor:", x_train_positive_tensor.size())

y_train_positive_tensor = torch.ones(10000, dtype = torch.long) ## this is y_train tensor for labels

print("y_train_positive_tensor:", y_train_positive_tensor.size())



#concat train

x_train = torch.cat((x_train_negative_tensor, x_train_positive_tensor), 0)

y_train = torch.cat((y_train_negative_tensor, y_train_positive_tensor), 0)

print("x_train:", x_train.size())

print("y_train:", y_train.size())





#read test negative

test_negative_path = r'/kaggle/input/lsifir/LSIFIR/Classification/Test/neg'

num_test_neg_img = 22050

test_negative_array = read_images(test_negative_path, num_test_neg_img)

x_test_negative_tensor = torch.from_numpy(test_negative_array[:18056,:]) ## numpy to tensor for using torch

print("x_test_negative_tensor:", x_test_negative_tensor.size())

y_test_negative_tensor = torch.zeros(18056, dtype = torch.long) ## this is y_train tensor for labels

print("y_test_negative_tensor:", y_test_negative_tensor.size())



#read test positive

test_positive_path = r'/kaggle/input/lsifir/LSIFIR/Classification/Test/pos'

num_test_pos_img = 5944

test_positive_array = read_images(test_positive_path, num_test_pos_img)

x_test_positive_tensor = torch.from_numpy(test_positive_array) ## numpy to tensor for using torch

print("x_test_positive_tensor:", x_test_positive_tensor.size())

y_test_positive_tensor = torch.ones(num_test_pos_img, dtype = torch.long) ## this is y_train tensor for labels

print("y_test_positive_tensor:", y_test_positive_tensor.size())



#concat test

x_test = torch.cat((x_test_negative_tensor, x_test_positive_tensor), 0)

y_test = torch.cat((y_test_negative_tensor, y_test_positive_tensor), 0)

print("x_test:", x_test.size())

print("y_test:", y_test.size())



#hyperparameter

num_epochs = 20 #100

num_classes = 2

batch_size = 100 #2000

lr = 0.00001



train = torch.utils.data.TensorDataset(x_train, y_train)

trainloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = "True")



test = torch.utils.data.TensorDataset(x_test, y_test)

testloader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = "True")
# x > conv > bn > relu > max pool > layer 1> layer 2> layer 3> avgpool > flatten > fc > output





def conv3x3(in_planes, output_planes, stride = 1):

    return nn.Conv2d(in_planes, output_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)



def conv1x1(in_planes, output_planes, stride = 1):

    return nn.Conv2d(in_planes, output_planes, kernel_size = 1, stride = stride, bias = False)



class BasicBlock(nn.Module):

    

    expansion = 1

    

    def __init__(self, inplanes, planes, stride = 1, downsample = None):

        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace = True)

        self.drop = nn.Dropout(0.9)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.stride = stride

        

    def forward(self, x):

        identity = x #shortcut

        

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.drop(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.drop(out) 

        

        if self.downsample is not None:

            identity = self.downsample(x)

            

        out = out + identity

        out = self.relu(out)

        return out

        

class ResNet(nn.Module):

    

    def __init__(self, block, layers, num_classes = num_classes):

        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace = True)

        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)

        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride = 1)

    

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(256*block.expansion, num_classes)

        

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)       

        

    def _make_layer(self, block, planes, blocks, stride = 1):

        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion:

            downsample = nn.Sequential(

                        conv1x1(self.inplanes, planes*block.expansion, stride),

                        nn.BatchNorm2d(planes*block.expansion))

            

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes*block.expansion

         

        for _ in range(1,blocks):

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

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        

        return x        
model = ResNet(BasicBlock, [2,2,2]).to(device)



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = lr)
# delete """(at the beginning and end) while you run the code



"""loss_list = []

train_acc = []

test_acc = []

use_gpu = True



total_step = len(trainloader)



for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(trainloader):

        

        images = images.view(images.size(0), 1, 64, 32)

        images = images.float()

        

        if use_gpu:

            if torch.cuda.is_available():

                images, labels = images.to(device), labels.to(device)

                

        outputs = model(images)

        

        loss = criterion(outputs, labels)

        

        #backward and optimization 

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if i % 2 == 0:

            print("epoch: {} {}/{}".format(epoch, i, total_step))

         

        # train 

        correct = 0

        total = 0

        with torch.no_grad():

            for data in trainloader:

                images, labels = data

                images = images.view(images.size(0), 1, 64, 32)

                images = images.float()

                

                if use_gpu:

                    if torch.cuda.is_available():

                        images, labels = images.to(device), labels.to(device)

                

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

                

        print("Accuracy train: %d %%" %(100*correct/total))

        train_acc.append(100*correct/total)

            

        # test 

        correct = 0

        total = 0

        with torch.no_grad():

            for data in testloader:

                images, labels = data

                images = images.view(images.size(0), 1, 64, 32)

                images = images.float()

                

                if use_gpu:

                    if torch.cuda.is_available():

                        images, labels = images.to(device), labels.to(device)

                

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

                

        print("Accuracy test: %d %%" %(100*correct/total))

        test_acc.append(100*correct/total)

        

        loss_list.append(loss.item())

        

        

            

            """
fig, ax1 = plt.subplots()

plt.plot(loss_list, label = "Loss", color = "blue")

ax2 = ax1.twinx()

ax2.plot(np.array(test_acc)/100, label = "Test Acc", color = "green")

ax2.plot(np.array(train_acc)/100, label = "Train Acc", color = "red")

ax1.legend()

ax2.legend()

ax1.set_xlabel("Epoch")

fig.tight_layout()

plt.title("Loss vs Test Accuracy")

plt.show()