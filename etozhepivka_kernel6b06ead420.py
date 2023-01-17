import numpy as np

import pandas as pd

import os

import matplotlib.image as mpimg



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms

import torchvision

from torch.utils.tensorboard import SummaryWriter



import warnings

warnings.filterwarnings("ignore")
data_dir = '../input'

train_dir = data_dir + '/train/train/'

test_dir = data_dir + '/test/test/'
labels = pd.read_csv("../input/train.csv")

labels.head()
class ImageData(Dataset):

    def __init__(self, df, data_dir, transform):

        super().__init__()

        self.df = df

        self.data_dir = data_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):       

        img_name = self.df.id[index]

        label = self.df.has_cactus[index]

        

        img_path = os.path.join(self.data_dir, img_name)

        image = mpimg.imread(img_path)

        image = self.transform(image)

        return image, label
epochs = 10

batch_size = 20

device = torch.device('cuda:0')
class Network(nn.Module): 

    def __init__(self):

        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)

        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)

        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512, 1024)

        self.fc2 = nn.Linear(1024, 2)



    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))

        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)))

        x = self.pool(x)

        x = x.view(x.shape[0],-1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x
data_transf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

train_data = ImageData(df = labels, data_dir = train_dir, transform = data_transf)

train_loader = DataLoader(dataset = train_data, batch_size = batch_size)
data_size = len(train_data)

validation_split = .2

split = int(np.floor(validation_split * data_size))

indices = list(range(data_size))

np.random.shuffle(indices)





train_indices, val_indices = indices[split:], indices[:split]
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

train_sampler = SubsetRandomSampler(train_indices)

val_sampler = SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 

                                           sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,

                                         sampler=val_sampler)
net = Network().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_func = nn.CrossEntropyLoss()
def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs):    

    loss_history = []

    train_history = []

    val_history = []

    for epoch in range(num_epochs):

        model.train() # Enter train mode

        

        loss_accum = 0

        correct_samples = 0

        total_samples = 0

        for i_step, (x, y) in enumerate(train_loader):

            prediction = model(x.to(device))    

            loss_value = loss(prediction, y.to(device))

            optimizer.zero_grad()

            loss_value.backward()

            optimizer.step()

            

            _, indices = torch.max(prediction, 1)

            correct_samples += torch.sum(indices == y.to(device))

            total_samples += y.shape[0]

            

            loss_accum += loss_value

    

        ave_loss = loss_accum / (i_step + 1)

        train_accuracy = float(correct_samples) / total_samples

        val_accuracy = compute_accuracy(model, val_loader)

        

        loss_history.append(float(ave_loss))

        train_history.append(train_accuracy)

        val_history.append(val_accuracy)

        

        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

        

    return loss_history, train_history, val_history, model
def compute_accuracy(model, loader):

    """

    Computes accuracy on the dataset wrapped in a loader

    

    Returns: accuracy as a float value between 0 and 1

    """

    correct_samples = 0

    total_samples = 0

    model.eval() # Evaluation mode

    for inputs, labels in loader:

        prediction = model(inputs.to(device))    

        _, indices = torch.max(prediction, 1)

        correct_samples += torch.sum(indices == labels.to(device))

        total_samples += labels.shape[0]

    validation_accuracy = float(correct_samples) / total_samples

        

    return validation_accuracy
loss_history, train_history, val_history, trained_model = train_model(net, train_loader, val_loader, loss_func, optimizer, 20)
submit = pd.read_csv('../input/sample_submission.csv')

test_data = ImageData(df = submit, data_dir = test_dir, transform = data_transf)

test_loader = DataLoader(dataset = test_data, shuffle=False)
net = trained_model

predict = []

for batch_i, (data, target) in enumerate(test_loader):

    data, target = data.to(device), target.to(device)

    output = net(data)

    

    _, pred = torch.max(output.data, 1)

    predict.append(int(pred))

    

submit['has_cactus'] = predict

submit.to_csv('submission.csv', index=False)
submit.head()