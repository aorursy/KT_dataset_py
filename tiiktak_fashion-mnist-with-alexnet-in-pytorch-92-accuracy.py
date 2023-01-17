import os

import torch

import torchvision

import numpy as np

import pandas as pd

import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt

import torch.nn.functional as F

from PIL import Image

from torchvision import transforms, datasets

from torch.utils.data import Dataset, DataLoader
# training batches of our network

EPOCHS = 10

# size of each batch

BATCH_SIZE = 512



DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



print(torch.__version__)

print(DEVICE)
train_csv = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test_csv = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')





print(train_csv.shape)

print(test_csv.shape)
print(train_csv.info())

print(train_csv.head())
class FashionDataset(Dataset):

    def __init__(self, data, transform=None):        

        self.fashion_MNIST = list(data.values)

        self.transform = transform

        

        label, image = [], []

        

        for i in self.fashion_MNIST:

            label.append(i[0])

            image.append(i[1:])

        self.labels = np.asarray(label)

        self.images = np.asarray(image).reshape(-1, 28, 28).astype('float32')

        

    def __len__(self):

        return len(self.images)

    

    def __getitem__(self, idx):

        label = self.labels[idx]

        image = self.images[idx]      

        

        if self.transform is not None:

            # transfrom the numpy array to PIL image before the transform function

            pil_image = Image.fromarray(np.uint8(image)) 

            image = self.transform(pil_image)

            

        return image, label
AlexTransform = transforms.Compose([

    transforms.Resize((227, 227)),

    transforms.ToTensor(),

    transforms.Normalize((0.1307,), (0.3081,))

])
train_loader = DataLoader(

    FashionDataset(train_csv, transform=AlexTransform), 

    batch_size=BATCH_SIZE, shuffle=True)



test_loader = DataLoader(

    FashionDataset(test_csv, transform=AlexTransform), 

    batch_size=BATCH_SIZE, shuffle=True)
# helper function to show an image

def matplotlib_imshow(img):

    img = img.mean(dim=0)

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    plt.imshow(npimg, cmap="Greys")



# get some random training images

dataiter = iter(train_loader)

images, labels = dataiter.next()



# creat grid of images

img_grid = torchvision.utils.make_grid(images[0])



# show images & labels

matplotlib_imshow(img_grid)

print(class_names[labels[0]])
class fasion_mnist_alexnet(nn.Module):  

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2)

        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(96, 256, 5, 1, 2),

            nn.ReLU(),

            nn.MaxPool2d(3, 2)

        )

        self.conv3 = nn.Sequential(

            nn.Conv2d(256, 384, 3, 1, 1),

            nn.ReLU()

        )

        self.conv4 = nn.Sequential(

            nn.Conv2d(384, 384, 3, 1, 1),

            nn.ReLU()

        )

        self.conv5 = nn.Sequential(

            nn.Conv2d(384, 256, 3, 1, 1),

            nn.ReLU(),

            nn.MaxPool2d(3, 2)

        )



        self.fc1 = nn.Linear(256 * 6 * 6, 4096)

        self.fc2 = nn.Linear(4096, 4096)

        self.fc3 = nn.Linear(4096, 10)



    def forward(self, x):

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        out = self.conv4(out)

        out = self.conv5(out)

        out = out.view(out.size(0), -1)



        out = F.relu(self.fc1(out))  # 256*6*6 -> 4096

        out = F.dropout(out, 0.5)

        out = F.relu(self.fc2(out))

        out = F.dropout(out, 0.5)

        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)



        return out

model = fasion_mnist_alexnet().to(DEVICE)

criterion = F.nll_loss

optimizer = optim.Adam(model.parameters())
def train(model, device, train_loader, optimer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        target = target.type(torch.LongTensor)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        if (batch_idx + 1) % 30 == 0:

            print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))
def test(model, device, test_loader):

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += criterion(output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()



        test_loss /= len(test_loader.dataset)  # loss之和除以data数量 -> mean

        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(

            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

        print('='*50)

for epoch in range(1, EPOCHS+1):

    train(model, DEVICE, train_loader, optimizer, epoch)

    test(model, DEVICE, test_loader)