#########################################                                
#
#         Anopheles Project
#        Detecting malaria in blood cells images
#       Done by Efi Eisenberg as a self-taught project
#
#    Everybody - It will be super cool if you find some way to make the network 
#   work better. Enjoy
#
####################################################################################


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

## Loading the DATA

GPU = True
VALIDATION_SIZE = 0.1
TRAIN_TEST_RATIO = 0.8
SHUFFLE_DATA_SPLIT = True
NUM_OF_EPOCH = 3
LR = 0.002
MOMENTUM = 0.9
RANDOM_SEED = np.random.randint(0, 2000000000) # 1984 is our experiment seed
print("random seed", RANDOM_SEED)

ROOT_DIR = "../input/cell_images/cell_images/"
## Preprocessing

transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.1395, 0.2531, 0.3396], [1, 1, 1])])

dataFrame_data = datasets.ImageFolder(root=ROOT_DIR,
                                      transform=transform)

df_size = len(dataFrame_data)
indices = list(range(df_size))
val_split = int(np.floor(VALIDATION_SIZE*df_size))
test_split = df_size - int(np.floor((df_size-val_split)*TRAIN_TEST_RATIO))

if SHUFFLE_DATA_SPLIT:
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)

val_indices, test_indices, train_indices = indices[:val_split], indices[val_split:test_split], indices[test_split:]

# Creating data samplers and loaders

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)  # SequentialSampler caused problems
valid_sampler = SubsetRandomSampler(val_indices)


train_loader = torch.utils.data.DataLoader(dataFrame_data,
                                           sampler=train_sampler,
                                           batch_size=2,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataFrame_data,
                                           sampler=test_sampler,
                                           batch_size=2,
                                           num_workers=2)

val_loader = torch.utils.data.DataLoader(dataFrame_data,
                                             sampler=valid_sampler,
                                             batch_size=2,
                                             num_workers=2)


classes = ('Parasitized', 'Uninfected')
## Defining the network

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 9, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(9, 16, 3)
        # an affine operation : y = Wx + b
        self.fc1 = nn.Linear(16*6*6, 288)
        self.fc2 = nn.Linear(288, 72)
        self.fc3 = nn.Linear(72, 2)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = x.view(-1, 16*6*6)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)

        return x
    
net = ConvNet()
## Training the network

print(net)
param = list(net.parameters())
print("len of param: ", len(param))

criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)

for epoch in range(NUM_OF_EPOCH):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the input
        inputs, labels = data

        # zero the parameters
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')
## Testing
print("Testing")

correct = 0
total = 0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

with torch.no_grad():
    for data in test_loader:
        images, labels = data

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        c = (predicted == labels).squeeze()
        if labels.shape[0] == 1:  # Batch of size 1 causes problems
            c = [c]
        c_size = len(c)

        for i in range(c_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print("Size of classes: ", class_total)

print('Accuracy of the network on %d test images: %d %%' % (len(test_loader),
    100 * correct / total))

for i in range(2):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


## Validation
print("\nValidation:")

correct = 0
total = 0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

with torch.no_grad():
    for data in val_loader:
        images, labels = data
        num_of_imgs = labels.shape[0]
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        c = (predicted == labels).squeeze()
        if labels.shape[0] == 1:  # Batch of size 1 causes problems
            c = [c]
        c_size = len(c)

        for i in range(c_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print("Size of classes: ", class_total)

print('Accuracy of the network on %d validation images: %d %%' % (len(val_loader), 100 * correct / total))

for i in range(2):
    print('Accuracy of validation %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))