import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import torch

import torchvision

import warnings

import shutil



%matplotlib inline

warnings.filterwarnings("ignore")
image_cat = pd.read_csv("../input/train.csv", low_memory=False, index_col="id").to_dict()["has_cactus"]
# ??torchvision.datasets.ImageFolder
from torchvision.datasets import DatasetFolder, ImageFolder

from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader



def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):

    images = []

    dir = os.path.expanduser(dir)

    

    for filename in os.listdir(dir):

        path = os.path.join(dir, filename)

        item = (path, image_cat[filename])

        images.append(item)

    return images



class CactusImageFolder(DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):                                                                                              

        self.root = root

        self.transform = transform

        self.target_transform = target_transform

        self.classes, self.class_to_idx = self._find_classes(self.root)

        self.samples = make_dataset(self.root, self.class_to_idx)

        self.loader = loader

        self.targets = [s[1] for s in self.samples]

        

    def _find_classes(self, dir):

        def f(x):

            if x in image_cat.keys():

                return "cactus" if image_cat[x] == 1 else "noncactus"

            else:

                return ""

        classes = list(set([ f(filename) for filename in os.listdir(dir)]))

        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx
from torchvision import transforms, datasets

from torch.utils.data.sampler import SubsetRandomSampler



BATCH = 10

data_transorm = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])



train_set = CactusImageFolder(root="../input/train/train", transform=data_transorm)

indices = list(range(0, len(image_cat)))

split = int(len(indices) * 0.2)

val_idx, train_idx = indices[:split], indices[split:]

val_sampler, train_sampler = SubsetRandomSampler(val_idx), SubsetRandomSampler(train_idx)



train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, sampler=val_sampler)



test_set = datasets.ImageFolder(root="../input/test", transform=data_transorm)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH)



classes = train_set.classes
def imshow(img):

    img = img / 2 + 0.5

    npimg = img.numpy()

    plt.figure(figsize=(5, 5))

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()

    

data_iter = iter(train_loader)

images, labels = data_iter.next()

imshow(torchvision.utils.make_grid(images, nrow=int(BATCH/2)))

print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH)))
import torch.nn as nn

import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

net = Net()
import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(3):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % 1000 == 0 and i != 0:    # print every 1000 mini-batches

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 1000))

            running_loss = 0.0



print('Finished Training')
test_iter = iter(test_loader)

images, _ = test_iter.next()

imshow(torchvision.utils.make_grid(images, nrow=int(BATCH/2)))



outputs = net(images)

_, predicted = torch.max(outputs, 1)



print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(BATCH)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

net.to(device)
from sklearn.metrics import classification_report

preds = []

score = 0



with torch.no_grad():

    for i, data in enumerate(val_loader):

        file_names = [os.path.basename(name) for name, _ in test_set.imgs[i*BATCH:i*BATCH+BATCH]]

        images, labels = data

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        preds += list(zip(file_names, predicted.numpy()))

        score += (labels == predicted).sum()

print((score.item() * 1.0) / len(preds))
torch.max(outputs.data, 1)
preds = []



with torch.no_grad():

    for i, data in enumerate(test_loader):

        file_names = [os.path.basename(name) for name, _ in test_loader.dataset.imgs[i*BATCH:i*BATCH+BATCH]]

        images, _ = data

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        preds += list(zip(file_names, predicted.numpy()))
!head ../input/sample_submission.csv
output = pd.DataFrame(preds, columns=["id", "has_cactus"])

output.to_csv("submission.csv", index=False)
output["has_cactus"].sum() / len(output)
rows, cols = 10, 5

i = 0

fig, ax = plt.subplots(nrows=rows, ncols=cols, squeeze=False, figsize=(40, 40))

fig.subplots_adjust(hspace=0.5, wspace=0.5)

for name, label in output.iloc[:50].values:    

    img = plt.imread(f"../input/test/test/{name}")

    row = int(i/cols)

    col = i%cols

    ax[row][col].imshow(img)

    ax[row][col].title.set_text(f"Cactus? {label == 1}")

    i += 1