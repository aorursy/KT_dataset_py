

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import torch

import torchvision

from torchvision import datasets, transforms

import torch.nn as nn

from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn.functional as F



import os

print(os.listdir("../input/cell_images/cell_images/"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_transforms = transforms.Compose([transforms.Resize((120, 120)),

                                       transforms.ColorJitter(0.05),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.RandomVerticalFlip(),

                                       transforms.RandomRotation(20),

                                       transforms.ToTensor(), 

                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

                                     ])
image_dir = "../input/cell_images/cell_images/"

train_set = datasets.ImageFolder(image_dir, transform=train_transforms)
test_size = 0.2



num_train = len(train_set)

indices = list(range(num_train))

np.random.shuffle(indices)



test_split = int(np.floor((test_size) * num_train))

test_index, train_index = indices[:test_split - 1], indices[test_split - 1:]



train_sampler = SubsetRandomSampler(train_index)

test_sampler = SubsetRandomSampler(test_index)



train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=104)

test_loader = DataLoader(train_set, sampler=test_sampler, batch_size=58)

print("Images in Test set: {}\nImages in Train set: {}".format(len(test_index), len(train_index)))
classes=['infected','uninfected']
def imshow(img):

    img = img / 2 + 0.5  # unnormalize

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    

images, labels = next(iter(train_loader))



fig = plt.figure(figsize=(25, 15))



for i in range(10):

    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[], title=classes[labels[i]])

    imshow(images[i])

plt.show()
class MosquitoNet(nn.Module):

    

    def __init__(self):

        super(MosquitoNet, self).__init__()

        

        self.layer1 = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        

        self.layer2 = nn.Sequential(

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        

        self.layer3 = nn.Sequential(

            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)

        )

            

        self.fc1 = nn.Linear(64*15*15, 512)

        self.fc2 = nn.Linear(512, 128)

        self.fc3 = nn.Linear(128, 2)

        self.drop = nn.Dropout2d(0.2)

        

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = out.view(out.size(0), -1)    # flatten out a input for Dense Layer

        out = self.fc1(out)

        out = F.relu(out)

        out = self.drop(out)

        out = self.fc2(out)

        out = F.relu(out)

        out = self.drop(out)

        out = self.fc3(out)

        

        return out

        
model = MosquitoNet()

model.to(device)

error = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(model)
num_epochs = 22

batch_size = 100 



for epoch in range(num_epochs):

    train_loss = 0.

    model.train()    # explictily stating the training

    print("beginning training:..........")

    for i, (images, labels) in enumerate(train_loader):

        images, labels = images.to(device), labels.to(device)

        train = images.view(-1, 3, 120, 120)

        outputs = model(train)

        

        optimizer.zero_grad()

        loss = error(outputs, labels)

        loss.backward()    #back-propagation

        optimizer.step()

        

        train_loss += loss.item() * batch_size

     

    print("Epoch: {}, Loss: {:.4f}".format(epoch + 1, train_loss / len(train_loader.dataset)))
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

correct = 0

total = 0

class_total = [0 for _ in range(2)]

class_correct = [0 for _ in range(2)]

batch_size = 58

# Lists used in Confusion Matrix

actual = []

predict = []



model.eval()    # explicitly stating the testing 

with torch.no_grad():

    for images, labels in test_loader:

        images, labels = images.to(device), labels.to (device)

        actual.append(labels.data.tolist())

        test = images.view(-1, 3, 120, 120)

        outputs = model(test)

        predicted = torch.max(outputs, 1)[1]

        predict.append(predicted.data.tolist())

        total += len(labels)

        correct += (predicted == labels).sum().item()

        # Calculating classwise accuracy

        c = (predicted == labels).squeeze()

        for i in range(batch_size):

            label = labels[i]

            class_correct[label] += c[i].item()

            class_total[label] += 1

        

print("Accuracy on the Test set: {:.2f}%".format(correct * 100 / total))

print()

for i in range(2):

    print("Accuracy of {} :  {:.2f}%   [{} / {}]".format(classes[i], class_correct[i] * 100 / class_total[i], 

                                           class_correct[i], class_total[i]))

    

dataiter = iter(test_loader)

images, labels = dataiter.next()





from PIL import Image

model.eval()

with torch.no_grad():

    image = ("/input/C101P62ThinF_IMG_20150923_170110_cell_22.png")

    image = image.to(device)

    test = image.view(-1,3,120,120)

    outputs = model(test)                                                              #left off, need input file string user interface, 2032 7/17

    predicted = torch.max(outputs, 1)[1]

    im = Image.open("input/C101P62ThinF_IMG_20150923_170110_cell_22.png")

    im.show()

    print(outputs, predicted)

    























print("testing finished. Thank you.")
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

import itertools



#flatten out 2D list into 1D

actual = list(itertools.chain.from_iterable(actual))

predict = list(itertools.chain.from_iterable(predict))

results = confusion_matrix(actual, predict)

print("Accuracy Score: ")

print("{:.4f}".format(accuracy_score(actual, predict)))

print()

print("Report: ")

print(classification_report(actual, predict))

print()

print("Confusion Matrix: ")

print(pd.DataFrame(results, columns=["Predicted No", "Predicted Yes"], index=["Actual No", "Actual Yes"]))
import seaborn as sns



sns.heatmap(results, cmap="magma", annot=True, fmt="d", cbar=False)