import torch
import os
from PIL import Image
from torch.utils.data import Dataset

class MalariaCells(Dataset):
    
    def __init__(self, root_dir, transform = None):
        
        super(MalariaCells, self).__init__()
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.img_info = []
        
        self.class_distribution = {'Parasitized':0, 'Uninfected':0}
        
        for dirname in os.listdir(self.root_dir):
                
            if(dirname != 'cell_images'):
                
                for filename in os.listdir(os.path.join(self.root_dir, dirname)):
                
                    if(not filename.endswith('.db')):
                        
                        label = 1 if dirname == 'Parasitized' else 0
                    
                        img_path = os.path.join(self.root_dir, dirname, filename)
                        
                        self.class_distribution[dirname]+=1
                        
                        self.img_info.append((img_path, label))
                        
                    
                
    def __len__(self):
        
        return len(self.img_info)
                
    def __getitem__(self, index):
        
        img_pth, label = self.img_info[index][0], self.img_info[index][1]
        
        image = Image.open(img_pth)
        
        label = torch.tensor(label, dtype = torch.float)
        
        if(self.transform is not None):
            
            image = self.transform(image)
            
        return image, label

import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

#at this point of time we load the dataset to only to do some EDA, later we will add additional transformations for train and test set separately
dataset = MalariaCells(root_dir = '/kaggle/input/cell-images-for-detecting-malaria/cell_images', transform = transforms.ToTensor())

print(dataset.class_distribution)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    

print("Parasitized images :")
show(dataset[0][0])
plt.show()

show(dataset[100][0])
plt.show()

show(dataset[200][0])
plt.show()

print("Uninfected images")
show(dataset[15000][0])
plt.show()

show(dataset[14000][0])
plt.show()

show(dataset[15050][0])
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

widths = []
heights = []

for image, label in dataset:
    
    widths.append(image.shape[2])
    heights.append(image.shape[1])
    
sns.distplot(widths, kde = True)
plt.show()

sns.distplot(heights, kde = True)
plt.show()
print("the maximum width is %s"%(max(widths), ))
print("the minimum width is %s"%(min(widths), ))
print("the maximum height is %s"%(max(heights), ))
print("the minimum height is %s"%(min(heights), ))
plt.figure(figsize = (12, 12))

sns.boxplot(widths)
plt.show()

plt.figure(figsize = (12, 12))

sns.boxplot(heights)
plt.show()
#constructing resnets to classify the 112X112 images

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 1, identity_downsample = None):
        
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = (3, 3), stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = (3, 3), stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.identity_downsample = identity_downsample
        
        
    def forward(self, x):
        
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if(self.identity_downsample is not None):
            
            identity = self.identity_downsample(identity)
            
        return F.relu(x + identity)
    
    
class ResNet(nn.Module):
    
    def __init__(self, image_channels, layers, num_classes):
        
        
        super(ResNet, self).__init__()
        
        #helper variable
        self.in_channels = 64
        
        #112X112
        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size = (5, 5), stride = 2, padding = 2)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        
        #56X56
        self.layer1 = self._make_layer(n = layers[0], out_channels = 64, stride = 1)
        
        #56X56
        self.layer2 = self._make_layer(n = layers[1], out_channels = 128, stride = 2)
        
        #28X28
        self.layer3 = self._make_layer(n = layers[2], out_channels = 256, stride = 2)
        
        #14X14
        self.layer4 = self._make_layer(n = layers[3], out_channels = 512, stride = 2)
        
        #7X7
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dense1 = nn.Linear(512, num_classes)
        
    
    def _make_layer(self, n, out_channels, stride):
        
        layer = []
        identity_downsample = None
        
        if(self.in_channels != out_channels or stride !=1):
            
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size = (1, 1), stride = stride, padding = 0), nn.BatchNorm2d(out_channels))
        
        layer.append(ResidualBlock(self.in_channels, out_channels, stride = stride, identity_downsample = identity_downsample))
        
        self.in_channels = out_channels
        
        for i in range(n-1):
            
            layer.append(ResidualBlock(out_channels, out_channels, stride = 1, identity_downsample = None))
            
        return nn.Sequential(*layer)
    
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pool1(x)
        
        x = x.view(x.shape[0], -1)
        
        x = self.dense1(x)
        
        return x
    
        
def ResNet18(image_channels, num_classes):
    
    return ResNet(image_channels, layers = [2, 2, 2, 2], num_classes = num_classes)
    

def ResNet34(image_channels, num_classes):

    return ResNet(image_channels, layers = [3, 4, 6, 3], num_classes = num_classes)
class MapDataset(Dataset):
    
    def __init__(self, dataset, transform = None):
        
        super(MapDataset, self).__init__()
        
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        
        return len(self.dataset)
        
    def __getitem__(self, index):
        
        return self.transform(self.dataset[index][0]), self.dataset[index][1]
from torch.utils.data import DataLoader
from torch.utils.data import random_split

transform1 = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(size = (112, 112), interpolation = 2),
                                transforms.RandomRotation(degrees = 45),
                                transforms.RandomHorizontalFlip(p = 0.5),
                                transforms.RandomVerticalFlip(p = 0.5),
                                transforms.ToTensor()
                                ])

transform2 = transforms.Compose([transforms.ToPILImage(),
                                 transforms.Resize(size = (112, 112), interpolation = 2),
                                 transforms.ToTensor()
                                ])

train_data, test_data = random_split(dataset, [22046, 5512])#20% of the training set

train_data = MapDataset(dataset = train_data, transform = transform1)
test_data = MapDataset(dataset = test_data, transform = transform2)

BATCH_SIZE = 32

train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True, num_workers = 2)
test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False, pin_memory = True, num_workers = 2)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = ResNet34(image_channels =3, num_classes = 1)
net = net.to(device)

print(device)
print(net)
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter

optimizer = torch.optim.SGD(net.parameters(), lr = 0.005)

loss_func = nn.BCEWithLogitsLoss()

writer = SummaryWriter(f'runs/resnet_34_metrics')

TRAIN_GLOBAL_STEP = 0
TEST_GLOBAL_STEP  = 0


def accuracy(predictions, labels):
    
    predicted_labels = torch.round(F.sigmoid(predictions))
    
    truth_values = (predicted_labels == labels)
    
    correct = truth_values.sum().item()
    
    return correct*100

def train(net, train_loader):
    
    global TRAIN_GLOBAL_STEP
    global BATCH_SIZE
    
    net.train()
    
    t = tqdm(enumerate(train_loader, 0), total = len(train_loader), desc = "TRAIN_EPOCH_LOSS :")
    
    running_loss = 0.0
    running_accuracy = 0.0
    
    for i, data in t:
        
        optimizer.zero_grad()
        
        inputs, labels = data[0].to(device), data[1].to(device)
        
        predictions = net(inputs)
        
        loss = loss_func(predictions, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_accuracy += accuracy(predictions, labels.unsqueeze(1))
        
        writer.add_scalar('train_batch_loss', loss.item(), global_step = TRAIN_GLOBAL_STEP)
        writer.add_scalar('train_batch_accuracy', accuracy(predictions, labels)/BATCH_SIZE, global_step = TRAIN_GLOBAL_STEP)
        
        TRAIN_GLOBAL_STEP += 1
        
    print("AVERAGE_TRAIN_LOSS: %s"%(running_loss/len(train_loader), ))
    print("AVERAGE_TRAIN_ACCURACY: %s"%(running_accuracy/len(train_loader.dataset), ))
        

def test(net, test_loader):
    
    global TEST_GLOBAL_STEP
    global BATCH_SIZE
    
    net.eval()
    
    t = tqdm(enumerate(test_loader, 0), total = len(test_loader), desc = "TEST_EPOCH_PROGRESS :")
    
    running_loss = 0.0
    running_accuracy = 0.0
    
    with torch.no_grad():
        
        for i, data in t:
            
            inputs, labels = data[0].to(device), data[1].to(device)
            
            predictions = net(inputs)
            
            loss = loss_func(predictions, labels.unsqueeze(1))
            
            running_loss += loss.item()
            
            running_accuracy += accuracy(predictions, labels.unsqueeze(1))
            
            writer.add_scalar('test_batch_loss', loss.item(), global_step = TEST_GLOBAL_STEP)
            writer.add_scalar('test_batch_accuracy', accuracy(predictions, labels)/BATCH_SIZE, global_step = TEST_GLOBAL_STEP)
            
            TEST_GLOBAL_STEP +=1
            
    print("AVERAGE_TEST_LOSS: %s"%(running_loss/len(test_loader), ))
    print("AVERAGE_TEST_ACCURACY: %s"%(running_accuracy/len(test_loader.dataset), ))
epochs = 20

for epoch in range(epochs):
    
    print("EPOCH [%s/%s]\n"%(epoch+1, epochs))
    
    train(net, train_loader)
    test(net, test_loader)
    
    print("------------------------------------------------------------------------------------------------")
#FN = (1-pred) * target


def FN_rate(net, test_loader):
    
    global TEST_GLOBAL_STEP
    global BATCH_SIZE
    
    net.eval()
    
    t = tqdm(enumerate(test_loader, 0), total = len(test_loader), desc = "TEST_EPOCH_PROGRESS :")
    
    running_fn_rate = 0.0
    
    with torch.no_grad():
        
        for i, data in t:
            
            inputs, labels = data[0].to(device), data[1].to(device)
            
            predictions = net(inputs)
            
            predicted_labels = torch.round(F.sigmoid(predictions))
            
            false_negatives = ((1-predicted_labels)*labels.unsqueeze(1)).sum().item()
            
            running_fn_rate += false_negatives
            
    print("FALSE_NEGATIVE_RATE: %s"%((running_fn_rate*100)/len(test_loader.dataset), ))
FN_rate(net, test_loader)